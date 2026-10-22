"""
Equational Drawing Tool
Converte uma foto (ou GIF animado) em um desenho feito so de equacoes
parametricas (curvas cubicas) e mostra tudo num plano cartesiano em uma
janela Tkinter. Com GIF, o original roda na velocidade real e o desenho
em equacoes acompanha frame a frame, enquanto as equacoes vao aparecendo.

Uso:  py main.py            (pede o caminho da foto no terminal)
      py main.py foto.gif   (ou passa direto como argumento)
"""
import os
import sys
import time
import tkinter as tk
import urllib.parse

import cv2
import numpy as np
from PIL import Image, ImageSequence, ImageTk

MAX_LADO = 400        # tamanho maximo da imagem processada (px)
MIN_CONTORNO = 15     # contornos menores que isso sao ignorados
PONTOS_POR_CURVA = 20 # quantos pontos do contorno cada equacao cobre
MAX_EQUACOES = 4000   # limite por frame pra janela nao travar
MAX_FRAMES = 300      # limite de frames lidos de um GIF

DURACAO_PADRAO = 100  # ms por frame quando o GIF nao informa
DURACAO_FOTO = 6.0    # imagem estatica: tempo (s) pra desenhar todas as equacoes
FRACAO_DESENHO = 0.8  # GIF: as curvas do frame ficam prontas em 80% da duracao dele
MS_TICK = 15          # intervalo entre passos do desenho (ms)
LADO_ORIGINAL = 260   # tamanho do preview do original (px)


def carregar_frames(caminho):
    """Retorna (frames BGR, frames originais PIL, duracoes em ms).
    Imagem normal = 1 frame, GIF = varios."""
    # Pillow em vez de cv2.imread: aceita acentos no caminho e GIF
    try:
        img = Image.open(caminho)
        originais, duracoes = [], []
        for _, f in zip(range(MAX_FRAMES), ImageSequence.Iterator(img)):
            originais.append(f.convert("RGB"))
            duracoes.append(max(20, f.info.get("duration") or DURACAO_PADRAO))
    except OSError as e:
        sys.exit(f"Nao consegui abrir a imagem: {caminho}\n{e}")
    frames = []
    for f in originais:
        bgr = cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        escala = MAX_LADO / max(h, w)
        if escala < 1:
            bgr = cv2.resize(bgr, (int(w * escala), int(h * escala)))
        frames.append(bgr)
    return frames, originais, duracoes


def extrair_contornos(img):
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cinza = cv2.GaussianBlur(cinza, (5, 5), 0)
    # limiares do Canny baseados na mediana da imagem
    med = np.median(cinza)
    bordas = cv2.Canny(cinza, int(max(0, 0.66 * med)), int(min(255, 1.33 * med)))
    contornos, _ = cv2.findContours(bordas, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return [c.reshape(-1, 2) for c in contornos if len(c) >= MIN_CONTORNO]


def ajustar_curvas(contornos, w, h):
    """Quebra cada contorno em pedacos e ajusta x(t), y(t) cubicos, t em [0,1]."""
    curvas = []
    for c in contornos:
        # pixel -> plano cartesiano (origem no centro, y pra cima)
        xs = c[:, 0] - w / 2
        ys = h / 2 - c[:, 1]
        for i in range(0, len(c) - 1, PONTOS_POR_CURVA - 1):
            px = xs[i:i + PONTOS_POR_CURVA]
            py = ys[i:i + PONTOS_POR_CURVA]
            if len(px) < 4:
                continue
            t = np.linspace(0, 1, len(px))
            cx = np.polyfit(t, px, 3)
            cy = np.polyfit(t, py, 3)
            curvas.append((cx, cy))
            if len(curvas) >= MAX_EQUACOES:
                return curvas
    return curvas


def poli_str(coef):
    """[a, b, c, d] -> 'a t³ + b t² + c t + d'"""
    termos = []
    for c, pot in zip(coef, ["t³", "t²", "t", ""]):
        if abs(c) < 0.005:
            continue
        sinal = "-" if c < 0 else "+"
        termos.append(f"{sinal} {abs(c):.2f}{pot}")
    if not termos:
        return "0"
    s = " ".join(termos)
    return s[2:] if s.startswith("+") else "-" + s[2:]


def equacao_str(n, cx, cy):
    return f"C{n}: x(t) = {poli_str(cx)}\n     y(t) = {poli_str(cy)}   0≤t≤1\n"


class Janela:
    def __init__(self, frames_curvas, originais, duracoes, w, h):
        self.frames = frames_curvas
        self.duracoes = duracoes
        self.gif = len(frames_curvas) > 1
        self.w, self.h = w, h
        self.root = tk.Tk()
        total = sum(len(c) for c in frames_curvas)
        self.root.title(f"Equational Drawing Tool - {len(frames_curvas)} frame(s), {total} equações")

        self.tam = 700
        self.zoom = (self.tam - 40) / max(w, h)

        self.canvas = tk.Canvas(self.root, width=self.tam, height=self.tam, bg="white")
        self.canvas.pack(side=tk.LEFT)

        lado = tk.Frame(self.root)
        lado.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # preview do original (o GIF roda aqui na velocidade real)
        self.fotos = []
        for o in originais:
            o = o.copy()
            o.thumbnail((LADO_ORIGINAL, LADO_ORIGINAL))
            self.fotos.append(ImageTk.PhotoImage(o, master=self.root))
        self.original = tk.Label(lado, bg="black")
        self.original.pack(side=tk.TOP, fill=tk.X)

        self.status = tk.Label(lado, font=("Consolas", 10, "bold"), anchor="w")
        self.status.pack(side=tk.TOP, fill=tk.X)
        scroll = tk.Scrollbar(lado)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.texto = tk.Text(lado, width=72, wrap=tk.NONE, font=("Consolas", 9),
                             yscrollcommand=scroll.set)
        self.texto.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.texto.tag_config("nova", foreground="#c0392b", font=("Consolas", 9, "bold"))
        scroll.config(command=self.texto.yview)

        self.t = np.linspace(0, 1, 10)
        self.desenhar_plano()
        self.job = self.job_frame = None
        self.ultimas = []
        self.atuais, self.antigas = [], []  # itens do canvas do frame atual / anterior
        self.proximo = time.perf_counter()  # relogio do GIF (instante em que o frame comeca)
        self.iniciar_frame(0)

    def tela(self, x, y):
        """plano cartesiano -> coordenadas do canvas"""
        return self.tam / 2 + x * self.zoom, self.tam / 2 - y * self.zoom

    def desenhar_plano(self):
        passo = 50  # unidades entre linhas da grade
        lim = int(max(self.w, self.h) / 2) + passo
        for v in range(-lim // passo * passo, lim + 1, passo):
            x0, y0 = self.tela(v, -lim)
            x1, y1 = self.tela(v, lim)
            self.canvas.create_line(x0, y0, x1, y1, fill="#e5e5e5")
            x0, y0 = self.tela(-lim, v)
            x1, y1 = self.tela(lim, v)
            self.canvas.create_line(x0, y0, x1, y1, fill="#e5e5e5")
            if v:
                tx, ty = self.tela(v, 0)
                self.canvas.create_text(tx, ty + 8, text=str(v), font=("Arial", 7), fill="#888")
                tx, ty = self.tela(0, v)
                self.canvas.create_text(tx - 12, ty, text=str(v), font=("Arial", 7), fill="#888")
        self.canvas.create_line(*self.tela(-lim, 0), *self.tela(lim, 0), fill="black", arrow=tk.LAST)
        self.canvas.create_line(*self.tela(0, -lim), *self.tela(0, lim), fill="black", arrow=tk.LAST)

    def iniciar_frame(self, f):
        """Mostra o frame f do original e comeca a desenhar as equacoes dele."""
        if self.job:
            self.root.after_cancel(self.job)
        self.f, self.i = f, 0
        self.inicio = time.perf_counter()
        self.ultimas = []
        self.original.config(image=self.fotos[f])
        # nao limpa o canvas: as curvas novas vao substituindo as do frame anterior,
        # assim o desenho fica sempre completo e anima junto com o GIF
        for item in self.atuais:
            self.canvas.itemconfig(item, fill="#2d70b3", width=1.5)
        self.antigas = self.atuais + self.antigas
        self.atuais = []
        self.texto.delete("1.0", tk.END)
        if self.gif:
            self.janela_desenho = self.duracoes[f] / 1000 * FRACAO_DESENHO
            self.agendar_proximo_frame()
        else:
            self.janela_desenho = DURACAO_FOTO
        self.desenhar_passo()

    def agendar_proximo_frame(self):
        # relogio absoluto: se o desenho atrasar, pula frames pra manter a velocidade real
        n = len(self.frames)
        f = self.f
        self.proximo += self.duracoes[f] / 1000
        agora = time.perf_counter()
        while self.proximo < agora:
            f = (f + 1) % n
            self.proximo += self.duracoes[f] / 1000
        prox = (f + 1) % n
        self.job_frame = self.root.after(max(1, int((self.proximo - agora) * 1000)),
                                         self.iniciar_frame, prox)

    def desenhar_passo(self):
        curvas = self.frames[self.f]
        # as curvas do passo anterior voltam pra cor normal
        for item in self.ultimas:
            self.canvas.itemconfig(item, fill="#2d70b3", width=1.5)
        self.texto.tag_remove("nova", "1.0", tk.END)

        # quantas curvas ja deviam estar desenhadas pelo tempo que passou
        progresso = (time.perf_counter() - self.inicio) / self.janela_desenho
        alvo = min(len(curvas), max(self.i + 1, int(len(curvas) * progresso)))

        self.ultimas, linhas = [], []
        for k in range(self.i, alvo):
            cx, cy = curvas[k]
            xs, ys = np.polyval(cx, self.t), np.polyval(cy, self.t)
            pts = [c for x, y in zip(xs, ys) for c in self.tela(x, y)]
            if self.antigas:
                self.canvas.delete(self.antigas.pop(0))
            item = self.canvas.create_line(*pts, fill="#e74c3c", width=2.5,
                                           smooth=True, tags="curva")
            self.ultimas.append(item)
            self.atuais.append(item)
            linhas.append(equacao_str(k + 1, cx, cy))
        self.i = alvo
        if self.i >= len(curvas):  # frame pronto: some o que sobrou do anterior
            for item in self.antigas:
                self.canvas.delete(item)
            self.antigas = []
        if linhas:
            inicio = self.texto.index("end-1c")
            self.texto.insert(tk.END, "".join(linhas))
            self.texto.tag_add("nova", inicio, "end-1c")
            self.texto.see(tk.END)

        prefixo = f"Frame {self.f + 1}/{len(self.frames)}  |  " if self.gif else ""
        self.status.config(text=f"{prefixo}Equação {self.i}/{len(curvas)}")

        if self.i < len(curvas):
            self.job = self.root.after(MS_TICK, self.desenhar_passo)
        else:
            self.job = None

    def fechar(self):
        # cancela os timers pendentes pra nao dar erro ao fechar no meio da animacao
        for job in (self.job, self.job_frame):
            if job:
                self.root.after_cancel(job)
        self.root.destroy()

    def rodar(self):
        self.root.protocol("WM_DELETE_WINDOW", self.fechar)
        self.root.mainloop()


def normalizar_caminho(bruto):
    """Limpa o que for colado: espacos, aspas, '& ' do PowerShell, file:///, %20, ~."""
    s = bruto.strip()
    if s.startswith("& "):           # arrastar arquivo pro PowerShell gera: & 'C:\...'
        s = s[2:].strip()
    while len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    s = s.strip("\"'")                # aspas soltas so de um lado
    if s.lower().startswith("file:"):  # caminho copiado do navegador
        s = urllib.parse.unquote(s[5:].lstrip("/"))
    return os.path.normpath(os.path.expandvars(os.path.expanduser(s)))


def pedir_caminho():
    while True:
        caminho = normalizar_caminho(input("Caminho da foto: "))
        if os.path.isfile(caminho):
            return caminho
        print(f"Arquivo nao encontrado: {caminho}\nTenta de novo (Ctrl+C pra sair).\n")


def main():
    if len(sys.argv) > 1:
        caminho = normalizar_caminho(" ".join(sys.argv[1:]))
    else:
        caminho = pedir_caminho()
    print(f"Usando: {caminho}")

    frames, originais, duracoes = carregar_frames(caminho)
    h, w = frames[0].shape[:2]
    print(f"Imagem {w}x{h}, {len(frames)} frame(s). Detectando bordas e ajustando equações...")
    frames_curvas = []
    for k, img in enumerate(frames, 1):
        frames_curvas.append(ajustar_curvas(extrair_contornos(img), w, h))
        print(f"\r  frame {k}/{len(frames)}: {len(frames_curvas[-1])} equações", end="", flush=True)
    print("\n")

    primeiras = frames_curvas[0]
    for n, (cx, cy) in enumerate(primeiras[:10], 1):
        print(equacao_str(n, cx, cy))
    if len(primeiras) > 10:
        print(f"... e mais {len(primeiras) - 10} (veja todas na janela)")

    Janela(frames_curvas, originais, duracoes, w, h).rodar()


if __name__ == "__main__":
    main()
