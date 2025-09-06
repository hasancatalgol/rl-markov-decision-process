#!/usr/bin/env python3
# GridWorld — Teacher Mode (Tabs UI v8: greedy tie set in CLI + wrapped Instructions + CSV greedy_set)
# Patched: theme toggle wired, unified tie tolerance, theming cleanups, CSV metadata, reproducible seed
from dataclasses import dataclass, field
from typing import Tuple, Dict, List
import os, json, random, csv, time, math

try:
    import pygame
    import pygame.freetype as ft
    from pygame import gfxdraw
except Exception:
    pygame = None
    ft = None

# Optional plotting (for bar charts like the video)
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# -------- UI/Env Settings --------
DEFAULT_AUTO_BUDGET = 40
STEP_DELAY_MS = 250
CELL_SIZE = 104
MARGIN = 6
SIDEBAR_W = 460

SHOW_POLICY_ARROW_ABOVE_AGENT = True
MAKE_MINI_DOCS = True
Q_EPS = 1e-6   # numerical tolerance for Q ties (unified)

# -------- Themes (dark default) --------
THEME_DARK = {
    "bg": (21, 24, 28), "grid": (55, 60, 67), "cell": (33, 37, 43),
    "wall": (90, 100, 112), "goal_pos": (36, 94, 64), "goal_neg": (140, 42, 48),
    "agent": (86, 140, 255), "panel": (28, 32, 38, 235), "text": (230, 235, 240),
    "muted": (150, 160, 170), "accent": (86, 140, 255), "button": (45, 50, 58),
    "button_active": (86, 140, 255), "cell_border": (58, 64, 72)
}
THEME_LIGHT = {
    "bg": (245, 247, 250), "grid": (220, 224, 230), "cell": (255, 255, 255),
    "wall": (80, 92, 104), "goal_pos": (36, 120, 70), "goal_neg": (160, 50, 55),
    "agent": (86, 140, 255), "panel": (255, 255, 255, 235), "text": (30, 35, 40),
    "muted": (110, 120, 130), "accent": (86, 140, 255), "button": (235, 238, 244),
    "button_active": (86, 140, 255), "cell_border": (210, 214, 220)
}

Action = int  # 0=up,1=right,2=down,3=left
ARROWS = {0:"↑", 1:"→", 2:"↓", 3:"←"}
DIRS   = {0:(0,-1), 1:(1,0), 2:(0,1), 3:(-1,0)}

# ---------- Text helpers ----------
def measure(font, text):
    if ft is not None:
        r = font.get_rect(text)
        return r.width, r.height
    img = font.render(text, True, (0,0,0))
    return img.get_width(), img.get_height()

def wrap(font, text, max_w):
    words = text.split(' ')
    lines, cur = [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if not t:
            continue
        wpx, _ = measure(font, t)
        if wpx <= max_w:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

@dataclass
class GridWorld:
    width: int = 5
    height: int = 5
    start: Tuple[int, int] = (0, 0)
    walls: Tuple[Tuple[int, int], ...] = ((1,1),(1,2),(3,3))
    step_reward: float = -0.04
    gamma: float = 0.95
    slip_prob: float = 0.0  # probability to slip to a random other action
    terminals: Dict[Tuple[int,int], float] = field(default_factory=lambda:{(4,4): 1.0})

    def states(self) -> List[Tuple[int, int]]:
        return [(x,y) for y in range(self.height) for x in range(self.width) if (x,y) not in self.walls]
    def is_terminal(self, s: Tuple[int, int]) -> bool:
        return s in self.terminals
    def actions(self, s: Tuple[int, int]) -> List[Action]:
        return [] if self.is_terminal(s) else [0,1,2,3]
    def _step_det(self, s: Tuple[int,int], a: Action) -> Tuple[Tuple[int,int], float]:
        if self.is_terminal(s): return s, 0.0
        x,y = s; dx,dy = DIRS[a]
        nx = min(max(0, x+dx), self.width-1); ny = min(max(0, y+dy), self.height-1)
        ns = (nx, ny)
        if ns in self.walls: ns = s
        r = self.terminals.get(ns, self.step_reward)
        return ns, r
    def step(self, s: Tuple[int,int], a: Action) -> Tuple[Tuple[int,int], float]:
        # stochastic slip for rollout
        if self.slip_prob>0 and random.random()<self.slip_prob:
            others=[b for b in [0,1,2,3] if b!=a]; a=random.choice(others)
        return self._step_det(s,a)

# -------- Planning utils (expected value with slip) --------
def expected_q(env: GridWorld, V: Dict[Tuple[int,int], float], s: Tuple[int,int], a: Action) -> float:
    p_main = 1.0 - env.slip_prob
    ns, r = env._step_det(s, a)
    val = p_main * (r + env.gamma * V[ns])
    if env.slip_prob>0.0:
        others=[b for b in [0,1,2,3] if b!=a]; p_each=env.slip_prob/len(others)
        for b in others:
            ns2, r2 = env._step_det(s, b)
            val += p_each * (r2 + env.gamma * V[ns2])
    return val

def greedy_set(env: GridWorld, V, s) -> List[Action]:
    if env.is_terminal(s):
        return []
    qs = [(a, expected_q(env, V, s, a)) for a in env.actions(s)]
    if not qs:
        return []
    max_q = max(q for _, q in qs)
    return [a for a, q in qs if q >= max_q - Q_EPS]  # tolerant tie via Q_EPS

def value_iteration(env: GridWorld, theta=1e-4, max_iter=1000):
    V = {s: 0.0 for s in env.states()}
    for _ in range(max_iter):
        delta = 0.0; newV = V.copy()
        for s in env.states():
            if env.is_terminal(s): newV[s]=0.0; continue
            newV[s] = max(expected_q(env, V, s, a) for a in env.actions(s))
            delta = max(delta, abs(newV[s]-V[s]))
        V = newV
        if delta<theta: break
    policy={}
    for s in env.states():
        if env.is_terminal(s): policy[s]=None; continue
        # choose one, but ties exist
        policy[s] = max(env.actions(s), key=lambda a: expected_q(env, V, s, a))
    return V, policy

def policy_iteration(env: GridWorld, max_iter=1000, eval_theta=1e-4):
    policy={s:(None if env.is_terminal(s) else random.choice(env.actions(s))) for s in env.states()}
    V={s:0.0 for s in env.states()}
    for _ in range(max_iter):
        # evaluate policy (iterative)
        while True:
            delta=0.0
            for s in env.states():
                if env.is_terminal(s): continue
                a=policy[s]; v_new=expected_q(env,V,s,a)
                delta=max(delta,abs(v_new-V[s])); V[s]=v_new
            if delta<eval_theta: break
        # improve policy
        stable=True
        for s in env.states():
            if env.is_terminal(s): continue
            old=policy[s]
            best=max(env.actions(s), key=lambda a: expected_q(env,V,s,a))
            policy[s]=best
            if best!=old: stable=False
        if stable: break
    return V, policy

# -------- CLI helpers --------
# ---- Visualization: bar plots for policy (greedy set), Q(s,·), and V(s)
#    Press 'b' in the app to pop up figures for the current state.

def plot_state_bars(env: GridWorld, V, policy, s):
    if plt is None:
        print("[NOTE] matplotlib not available. pip install matplotlib to view bar charts.")
        return
    if env.is_terminal(s):
        print("[NOTE] Terminal state — no action bars to show.")
        return
    # Compute greedy set and Qs
    acts = env.actions(s)
    qs = [expected_q(env, V, s, a) for a in acts]
    max_q = max(qs) if qs else 0.0
    ties = [a for a, q in zip(acts, qs) if q >= max_q - Q_EPS]
    # Policy bars: equal mass over greedy ties (deterministic greedy => 1 at the best action)
    pi = [ (1.0/len(ties)) if a in ties else 0.0 for a in acts ] if ties else [0.0]*len(acts)
    # 1) Policy bars
    plt.figure()
    plt.bar(range(len(acts)), pi)
    plt.title(f"Policy π(a|s) at s={s}")
    plt.xlabel("action (0=↑,1=→,2=↓,3=←)")
    plt.ylabel("probability")
    plt.xticks(range(len(acts)), acts)
    # 2) Q bars
    plt.figure()
    plt.bar(range(len(acts)), qs)
    plt.title(f"Action-value Q(s,a) at s={s}")
    plt.xlabel("action (0=↑,1=→,2=↓,3=←)")
    plt.ylabel("Q(s,a)")
    plt.xticks(range(len(acts)), acts)
    # 3) Print V(s)
    print(f"V(s={s}) ≈ {V.get(s,0.0):+.6f}")
    plt.show()
def render_policy_grid(env: GridWorld, policy: Dict[tuple, Action]) -> str:
    lines=[]
    for y in range(env.height):
        row=[]
        for x in range(env.width):
            s=(x,y)
            if s in env.walls: row.append("##")
            elif s in env.terminals: row.append("G+" if env.terminals[s]>0 else "G-")
            elif s==env.start: row.append("S ")
            else: row.append(ARROWS.get(policy.get(s), "· "))
        lines.append(" ".join(row))
    return "\n".join(lines)

def stringify_actions(actions: List[Action]) -> str:
    return "{" + ", ".join(ARROWS[a] for a in actions) + "}"

def print_q_table_for_state(env: GridWorld, V, s, policy):
    if env.is_terminal(s): return []
    print(f"State s={s}: V(s)≈{V.get(s,0.0):+.3f}")
    qs = []
    for a in env.actions(s):
        ns_det, r_det = env._step_det(s, a)
        q = expected_q(env, V, s, a)
        qs.append((a,q))
        tag = " ← greedy" if policy.get(s)==a else ""
        print(f"  {ARROWS[a]}  ns_det={ns_det}  r_det={r_det:+.2f}  |  Q_exp≈{q:+.3f}{tag}")
    # greedy tie set
    max_q = max(q for _, q in qs) if qs else 0.0
    ties = [a for a, q in qs if q >= max_q - Q_EPS]
    if len(ties) > 1:
        print(f"  Greedy tie set: {stringify_actions(ties)} (|Δ| ≤ {Q_EPS})")
    return ties

# ---------- Minimal UI widgets ----------
class Button:
    def __init__(self, rect, label, value, accent=False):
        self.rect=pygame.Rect(rect); self.label=label; self.value=value; self.accent=accent; self.active=False
    def draw(self, surface, fonts, T):
        color=T["button_active"] if (self.active or self.accent) else T["button"]
        pygame.draw.rect(surface,color,self.rect,border_radius=14)
        if ft is not None:
            w=fonts["ui"].get_rect(self.label).width
            fonts["ui"].render_to(surface,(self.rect.centerx-w//2,self.rect.y+12),self.label,(255,255,255) if (self.active or self.accent) else T["text"])
        else:
            img=fonts["ui"].render(self.label,True,(255,255,255) if (self.active or self.accent) else T["text"])
            surface.blit(img,(self.rect.centerx-img.get_width()//2,self.rect.y+(self.rect.height-img.get_height())//2))
    def handle(self,pos): return self.value if self.rect.collidepoint(pos) else None

class Toggle:
    def __init__(self, rect, value=False): self.rect=pygame.Rect(rect); self.value=value
    def draw(self,surface,fonts,T,label,x_label):
        r=self.rect.height//2
        pygame.draw.rect(surface, T["button_active"] if self.value else T["button"], self.rect, border_radius=r)
        knob_r=r-3; knob_x=self.rect.x+(self.rect.width-self.rect.height)+3 if self.value else self.rect.x+3
        pygame.draw.circle(surface,(240,240,240),(knob_x+knob_r,self.rect.y+3+knob_r),knob_r)
        if ft is not None: fonts["ui"].render_to(surface,(x_label,self.rect.y+self.rect.height//2-10),label,T["text"])
        else: img=fonts["ui"].render(label,True,T["text"]) ; surface.blit(img,(x_label,self.rect.y+self.rect.height//2-10))
    def handle(self,pos): 
        if self.rect.collidepoint(pos): self.value=not self.value; return self.value
        return None

class Slider:
    def __init__(self, rect, min_v, max_v, value):
        self.rect=pygame.Rect(rect); self.min=min_v; self.max=max_v; self.value=value; self.knob_w=14
    def draw(self,surface,fonts,T,label,x,width,suffix=""):
        self.rect.x=x; self.rect.width=width
        title=f"{label}: {int(self.value)}{suffix}"
        if ft is not None: fonts["ui"].render_to(surface,(x,self.rect.y-26),title,T["text"])
        else: img=fonts["ui"].render(title,True,T["text"]) ; surface.blit(img,(x,self.rect.y-26))
        pygame.draw.rect(surface,T["grid"],self.rect,border_radius=8)
        t=(self.value-self.min)/(self.max-self.min); kx=int(self.rect.x+t*(self.rect.width-self.knob_w))
        pygame.draw.rect(surface,T["button_active"],pygame.Rect(kx,self.rect.y,self.knob_w,self.rect.height),border_radius=8)
    def handle(self,pos):
        if self.rect.collidepoint(pos):
            rel=(pos[0]-self.rect.x)/max(1,(self.rect.width-self.knob_w))
            self.value=max(self.min,min(self.max,self.min+rel*(self.max-self.min)))
            return int(self.value)
        return None

# ---------- Visualizer ----------
class Visualizer:
    def __init__(self, env, cell=CELL_SIZE, margin=MARGIN, sidebar_w=SIDEBAR_W):
        if pygame is None: raise RuntimeError("pygame not installed. pip install pygame")
        pygame.init()
        if ft is not None: ft.init()
        self.env=env; self.theme=THEME_DARK
        self.cell=cell; self.margin=margin; self.sidebar_w=sidebar_w
        self.grid_w=env.width*cell+(env.width+1)*margin
        self.grid_h=env.height*cell+(env.height+1)*margin
        self.screen=pygame.display.set_mode((self.grid_w+sidebar_w,self.grid_h))
        pygame.display.set_caption("GridWorld — Teacher Mode (v8 Ties+Wrap)")

        # fonts
        if ft is not None:
            self.font_title=ft.SysFont(None,26); self.font_ui=ft.SysFont(None,20); self.font_small=ft.SysFont(None,14)
        else:
            self.font_title=pygame.font.SysFont(None,26); self.font_ui=pygame.font.SysFont(None,20); self.font_small=pygame.font.SysFont(None,14)
        self.fonts={"title":self.font_title,"ui":self.font_ui,"small":self.font_small}

        # episode / mode
        self.auto=False; self.auto_budget=DEFAULT_AUTO_BUDGET; self.auto_max=DEFAULT_AUTO_BUDGET
        self.s=self.env.start; self.G=0.0; self.pow_gamma=1.0; self.t=0
        self.header_printed=False
        self.episode_rows=[]  # for CSV

        self.V_vi,self.pi_vi=value_iteration(self.env)
        self.V_pi,self.pi_pi=policy_iteration(self.env)
        self.mode="VI"; self.policy=self.pi_vi

        # tabs & controls
        self.tabs=[Button((self.grid_w+20,12,110,36),"Modes","Modes"),
                   Button((self.grid_w+140,12,120,36),"Settings","Settings"),
                   Button((self.grid_w+270,12,140,36),"Instructions","Instructions")]
        self.active_tab="Modes"
        x=self.grid_w+24; w=self.sidebar_w-48; y=86
        self.btn_vi=Button((x,y,w,44),"Value Iteration (VI)","VI"); y+=56
        self.btn_pi=Button((x,y,w,44),"Policy Iteration (PI)","PI"); y+=56
        self.btn_rand=Button((x,y,w,44),"Random actions","RANDOM"); y+=66
        self.btn_auto=Button((x,y,w,40),"Start Auto Move","AUTO",accent=True)
        # settings
        y=100
        self.toggle_theme=Toggle((x,y,76,32),value=True); y+=70
        self.slider_budget=Slider((x,y,w,14),5,200,self.auto_max); y+=60
        self.slider_slip=Slider((x,y,w,14),0,40,int(self.env.slip_prob*100)); y+=60
        self.toggle_trap=Toggle((x,y,76,32),value=False)
        self.clock=pygame.time.Clock(); self.next_step_time_ms=0

    # ------ helpers ------
    def update_highlights(self):
        self.btn_vi.active=(self.mode=='VI'); self.btn_pi.active=(self.mode=='PI'); self.btn_rand.active=(self.mode=='RANDOM')
        self.btn_auto.accent=self.auto; self.btn_auto.label="Pause Auto Move" if self.auto else "Start Auto Move"

    def reset(self):
        self._finish_episode_csv()  # close previous episode if any
        self.s=self.env.start; self.G=0.0; self.pow_gamma=1.0; self.t=0
        self.auto=False; self.auto_budget=self.auto_max; self.next_step_time_ms=0
        self.header_printed=False
        self.episode_rows=[]

    def _episode_header_cli(self):
        V=self._current_V(); pol=self.policy if self.mode in ('VI','PI') else {s:None for s in self.env.states()}
        print("\n────────────────────────────────────────────────────────")
        print(f"Episode start — Current Mode: {self.mode}")
        print(f"γ = {self.env.gamma} | step_reward={self.env.step_reward:+.2f} | slip={self.env.slip_prob*100:.0f}% | terminals={self.env.terminals}")
        print("Legend: r_t (immediate), γ^t (discount weight), contrib_t=r_t*γ^t, G_t (running discounted).")
        print("\nPolicy π(s):\n" + render_policy_grid(self.env, pol))
        print("────────────────────────────────────────────────────────")

    def _finish_episode_csv(self):
        if not self.episode_rows:
            return
        os.makedirs("docs", exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        fname = f"docs/episode_log_{ts}.csv"
        with open(fname, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["t","mode","s","a","ns","r_t","gamma_pow","contrib_t","G_t","V_s","Q_a","greedy_set"])
            for row in self.episode_rows:
                writer.writerow(row)
        # metadata sidecar for reproducibility
        terms = [{"state": [int(s[0]), int(s[1])], "reward": float(r)} for s, r in self.env.terminals.items()]
        meta = {
            "gamma": self.env.gamma,
            "step_reward": self.env.step_reward,
            "slip_prob": self.env.slip_prob,
            "terminals": terms,
            "width": self.env.width,
            "height": self.env.height,
            "start": self.env.start,
            "walls": self.env.walls,
            "mode": self.mode,
            "auto_max": self.auto_max,
            "theme": "dark" if self.theme is THEME_DARK else "light",
            "Q_EPS": Q_EPS,
        }
        with open(fname.replace('.csv', '.meta.json'), 'w', encoding='utf-8') as jf:
            json.dump(meta, jf, indent=2)
        print(f"[SAVED] Episode CSV -> {fname}")

    def set_mode(self, mode):
        self.mode=mode; self.policy=self.pi_vi if mode=='VI' else self.pi_pi if mode=='PI' else {}
        self.reset()
        self._episode_header_cli()

    def apply_settings(self):
        # apply env toggles
        self.env.slip_prob = self.slider_slip.value/100.0
        if self.toggle_trap.value: self.env.terminals[(0,4)] = -1.0
        else: self.env.terminals.pop((0,4), None)
        # theme toggle
        self.theme = THEME_DARK if self.toggle_theme.value else THEME_LIGHT
        # recompute V/π after changes
        self.V_vi,self.pi_vi=value_iteration(self.env)
        self.V_pi,self.pi_pi=policy_iteration(self.env)
        self.policy=self.pi_vi if self.mode=='VI' else self.pi_pi if self.mode=='PI' else {}
        self.reset()
        self._episode_header_cli()

    def _current_V(self):
        return self.V_vi if self.mode=='VI' else self.V_pi if self.mode=='PI' else self.V_vi

    # ------ CLI per-step math ------
    def step_log(self, a, ns, r):
        V = self._current_V()
        pol = self.policy if self.mode in ('VI','PI') else {}
        # Q-table + ties for current state
        ties = []
        if not self.env.is_terminal(self.s):
            print(f"State s={self.s}: V(s)≈{V.get(self.s,0.0):+.3f}")
            qs = []
            for act in self.env.actions(self.s):
                ns_det, r_det = self.env._step_det(self.s, act)
                q = expected_q(self.env, V, self.s, act)
                qs.append((act,q))
                tag = " ← greedy" if pol.get(self.s)==act else ""
                print(f"  {ARROWS[act]}  ns_det={ns_det}  r_det={r_det:+.2f}  |  Q_exp≈{q:+.3f}{tag}")
            max_q = max(q for _, q in qs) if qs else 0.0
            ties = [act for act, q in qs if q >= max_q - Q_EPS]
            if len(ties) > 1:
                print(f"  Greedy tie set: {stringify_actions(ties)} (|Δ| ≤ {Q_EPS})")
        contrib = self.pow_gamma * r
        print(f"t={self.t:02d}: s={self.s}  a={ARROWS[a]}  ->  s'={ns}{' (TERMINAL)' if self.env.is_terminal(ns) else ''} | r_t={r:+.2f}  γ^{self.t}={self.pow_gamma:.3f}  contrib_t={contrib:+.3f}  G_t={self.G+contrib:+.3f}")
        # store for CSV
        q_chosen = expected_q(self.env, V, self.s, a)
        self.episode_rows.append([self.t, self.mode, str(self.s), ARROWS[a], str(ns), f"{r:+.2f}", f"{self.pow_gamma:.6f}", f"{contrib:+.6f}", f"{self.G+contrib:+.6f}", f"{V.get(self.s,0.0):+.6f}", f"{q_chosen:+.6f}", stringify_actions(ties)])

    # ------ Pygame drawing ------
    def draw_grid(self):
        T=self.theme; self.screen.fill(T["bg"])
        for y in range(self.env.height):
            for x in range(self.env.width):
                rx=x*self.cell+(x+1)*self.margin; ry=y*self.cell+(y+1)*self.margin
                rect=pygame.Rect(rx,ry,self.cell,self.cell)
                pygame.draw.rect(self.screen,T["cell"],rect,border_radius=10)
                pygame.draw.rect(self.screen,T["cell_border"],rect,width=1,border_radius=10)
                if (x,y) in self.env.walls: pygame.draw.rect(self.screen,T["wall"],rect,border_radius=10)
                if (x,y) in self.env.terminals:
                    col = T["goal_pos"] if self.env.terminals[(x,y)]>0 else T["goal_neg"]
                    pygame.draw.rect(self.screen,col,rect,border_radius=10)
        # start border
        sx,sy=self.env.start
        srx=sx*self.cell+(sx+1)*self.margin; sry=sy*self.cell+(sy+1)*self.margin
        pygame.draw.rect(self.screen,self.theme["accent"],pygame.Rect(srx,sry,self.cell,self.cell),width=3,border_radius=10)
        # agent
        ax,ay=self.s; cx=ax*self.cell+(ax+1)*self.margin+self.cell//2; cy=ay*self.cell+(ay+1)*self.margin+self.cell//2
        r=max(12,self.cell//3); gfxdraw.filled_circle(self.screen,cx,cy,r,self.theme["agent"]); gfxdraw.aacircle(self.screen,cx,cy,r,self.theme["agent"])
        if SHOW_POLICY_ARROW_ABOVE_AGENT and self.mode in ('VI','PI') and not self.env.is_terminal(self.s):
            a=self.policy.get(self.s) 
            if a is not None:
                if ft is not None: self.fonts["title"].render_to(self.screen,(cx-8,cy-r-28),ARROWS[a],self.theme["accent"])
                else: img=self.fonts["title"].render(ARROWS[a],True,self.theme["accent"]) ; self.screen.blit(img,(cx-8,cy-r-28))

    def draw_sidebar(self):
        px=self.grid_w+10; py=10
        panel=pygame.Surface((self.sidebar_w-20,self.grid_h-20), pygame.SRCALPHA)
        pygame.draw.rect(panel,self.theme["panel"],panel.get_rect(),border_radius=16)
        self.screen.blit(panel,(px,py))

        # tabs
        for b in self.tabs:
            b.active=(b.value==self.active_tab); b.draw(self.screen,self.fonts,self.theme)

        x=self.grid_w+24; w=self.sidebar_w-48; y=86
        self.update_highlights()
        if self.active_tab=="Modes":
            for btn in (self.btn_vi,self.btn_pi,self.btn_rand,self.btn_auto):
                btn.rect.topleft=(x,y); btn.rect.width=w; btn.draw(self.screen,self.fonts,self.theme); y+=56 if btn!=self.btn_auto else 64
            for i,t in enumerate([f"Pos: {self.s}", f"G_t (discounted): {self.G:+.2f}", f"Auto budget: {self.auto_budget}/{self.auto_max}", f"Mode: {self.mode}"]):
                if ft is not None: self.fonts["ui"].render_to(self.screen,(x,y+i*26),t,self.theme["text"])
                else: img=self.fonts["ui"].render(t,True,self.theme["text"]) ; self.screen.blit(img,(x,y+i*26))
        elif self.active_tab=="Settings":
            self.toggle_theme.rect.topleft=(x,y); self.toggle_theme.draw(self.screen,self.fonts,self.theme,"Dark theme",x+96); y+=60
            self.slider_budget.rect.y=y; self.slider_budget.draw(self.screen,self.fonts,self.theme,"Auto-step budget",x,w); y+=60
            self.slider_slip.rect.y=y; self.slider_slip.draw(self.screen,self.fonts,self.theme,"Slip probability",x,w,suffix="%"); y+=60
            self.toggle_trap.rect.topleft=(x,y); self.toggle_trap.draw(self.screen,self.fonts,self.theme,"Enable trap goal at (0,4) with reward -1.0",x+96); y+=40
            apply_btn = Button((x,y,180,36),"Apply settings","APPLY",accent=True)
            apply_btn.draw(self.screen,self.fonts,self.theme)
            self._apply_button=apply_btn
        else:
            # Wrapped instructions
            paragraphs=[
                "Modes:",
                "• VI: Bellman optimality updates on V(s); act greedily w.r.t. V.",
                "• PI: alternate policy evaluation/improvement until policy stabilizes.",
                "• Random: no planning; uniform random action each step.",
                "",
                "CLI details printed each step: deterministic next state ns_det, immediate reward r_det for each action, expected Q(s,a) under slip, greedy tie set (all maximizers), then r_t, γ^t, contribution, and running discounted G_t.",
                "Episodes are saved as CSV in docs/ with columns including the greedy_set.",]
            yy=y; max_w=w
            for para in paragraphs:
                for line in wrap(self.fonts["ui"], para, max_w):
                    if ft is not None: self.fonts["ui"].render_to(self.screen,(x,yy),line,self.theme["text"])
                    else: img=self.fonts["ui"].render(line,True,self.theme["text"]) ; self.screen.blit(img,(x,yy))
                    yy+=22
                yy+=4

    # ------ interaction ------
    def handle_mouse(self,pos):
        for b in self.tabs:
            val=b.handle(pos)
            if val is not None: self.active_tab=val; return
        if self.active_tab=="Modes":
            for b in (self.btn_vi,self.btn_pi,self.btn_rand,self.btn_auto):
                val=b.handle(pos)
                if val is not None:
                    if val=="AUTO":
                        self.auto=not self.auto
                        if self.auto: 
                            self.auto_budget=self.auto_max; 
                            self.next_step_time_ms=pygame.time.get_ticks()+STEP_DELAY_MS
                            if not self.header_printed:
                                self._episode_header_cli(); self.header_printed=True
                        return
                    self.set_mode(val); return
        elif self.active_tab=="Settings":
            if self.toggle_theme.handle(pos) is not None: pass
            if self.slider_budget.handle(pos) is not None: self.auto_max=int(self.slider_budget.value); self.auto_budget=min(self.auto_budget,self.auto_max)
            if self.slider_slip.handle(pos) is not None: pass
            if self.toggle_trap.handle(pos) is not None: pass
            if hasattr(self,"_apply_button"):
                if self._apply_button.handle(pos)=="APPLY": self.apply_settings()

    def take_step(self):
        if not self.header_printed:
            self._episode_header_cli(); self.header_printed=True
        if self.env.is_terminal(self.s): 
            print("[PAUSE] At terminal."); return
        if self.mode in ('VI','PI'):
            a=self.policy.get(self.s, random.choice(self.env.actions(self.s)))
        else:
            a=random.choice(self.env.actions(self.s))
        ns,r=self.env.step(self.s,a)
        # CLI + CSV
        self.step_log(a, ns, r)
        # update accumulators
        self.G += self.pow_gamma * r
        self.pow_gamma *= self.env.gamma
        self.t += 1
        self.s=ns
        if self.env.is_terminal(self.s):
            print(f"Episode summary: steps={self.t}, discounted return G={self.G:+.3f}")
            self._finish_episode_csv()

    def loop(self):
        self.reset()
        running=True
        while running:
            now=pygame.time.get_ticks()
            for event in pygame.event.get():
                if event.type==pygame.QUIT: running=False
                if event.type==pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE): running=False
                    elif event.key==pygame.K_r: self.reset()
                    elif event.key==pygame.K_s: self.take_step()
                    elif event.key==pygame.K_b:
                        # bar plots for current state (like the video)
                        V = self._current_V()
                        pol = self.policy if self.mode in ('VI','PI') else {}
                        plot_and_save_state_bars(self.env, V, self.s, mode=self.mode)
                if event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
                    self.handle_mouse(event.pos)
            if self.auto and not self.env.is_terminal(self.s) and self.auto_budget>0 and now>=self.next_step_time_ms:
                self.take_step(); self.auto_budget-=1; self.next_step_time_ms=now+STEP_DELAY_MS
                if self.env.is_terminal(self.s): self.auto=False
                elif self.auto_budget==0: self.auto=False
            self.draw_grid(); self.draw_sidebar(); pygame.display.flip(); self.clock.tick(60)

def plot_and_save_state_bars(env: GridWorld, V, s, mode="VI"):
    """Make policy/Q bar charts for the current state and save PNGs under docs/plots/.
    Uses the greedy-tie-set over Q(s,·) induced by V for π(a|s) bars (like in the video)."""
    if plt is None:
        print("[NOTE] matplotlib not available. pip install matplotlib to view bar charts.")
        return
    if env.is_terminal(s):
        print("[NOTE] Terminal state — no action bars to show.")
        return
    acts = env.actions(s)
    qs = [expected_q(env, V, s, a) for a in acts]
    max_q = max(qs) if qs else 0.0
    ties = [a for a, q in zip(acts, qs) if q >= max_q - Q_EPS]
    pi = [ (1.0/len(ties)) if a in ties else 0.0 for a in acts ] if ties else [0.0]*len(acts)
    x = list(range(len(acts)))
    labels = [f"{a}:{ARROWS[a]}" for a in acts]
    fig1 = plt.figure()
    plt.bar(x, pi)
    plt.title(f"Policy π(a|s) at s={s} (mode={mode})")
    plt.xlabel("action")
    plt.ylabel("probability")
    plt.xticks(x, labels)
    fig2 = plt.figure()
    plt.bar(x, qs)
    plt.title(f"Action-value Q(s,a) at s={s} (mode={mode})")
    plt.xlabel("action")
    plt.ylabel("Q(s,a)")
    plt.xticks(x, labels)
    # Save
    os.makedirs("docs/plots", exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    base = f"s={s[0]},{s[1]}_{mode}_{ts}"
    f1 = os.path.join("docs","plots", f"policy_{base}.png")
    f2 = os.path.join("docs","plots", f"q_{base}.png")
    fig1.savefig(f1, bbox_inches="tight")
    fig2.savefig(f2, bbox_inches="tight")
    print(f"[SAVED] {f1}")
    print(f"[SAVED] {f2}")
    v_s = V.get(s, 0.0)
    print(f"V(s={s}) ≈ {v_s:+.6f}")
    plt.show()

def main():
    # reproducibility for demos
    random.seed(0)
    env=GridWorld()
    if MAKE_MINI_DOCS: os.makedirs("docs", exist_ok=True)
    if pygame is None:
        print("[NOTE] pygame not available here. Run locally:")
        print("pip install pygame")
        print("python grid_mdp_teacher_mode_ui_tabs_v8_ties_and_wrap.py")
    else:
        Visualizer(env, cell=CELL_SIZE, margin=MARGIN, sidebar_w=SIDEBAR_W).loop()

if __name__ == "__main__":
    main()  
