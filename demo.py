"""Offline, native desktop showcase. Run python demo.py [recording.c3d]."""
import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from motion import Recording

BG, PANEL, FG, MUTED = '#ffffff', '#f3f6fb', '#172b46', '#586b82'
CYAN, GOLD = '#007f89', '#b86610'
COLORS = {'Happy':'#ae7400','Sad':'#3975d5','Angry':'#c54661','Neutral':'#72758a',
          'Non-protective':'#008779','Protective':'#9b59b6'}
CHAINS = [
    ['HeadFront','HeadLeft','HeadTop','HeadRight','HeadFront'],
    ['HeadTop','BackTop','WaistCBack'],
    ['LShoulderTop','ChestTop','RShoulderTop'],
    ['ChestTop','ChestLow','WaistLFront','WaistRFront','ChestLow'],
    ['WaistLFront','WaistLBack','WaistCBack','WaistRBack','WaistRFront','WaistLFront'],
]
for side in 'LR':
    CHAINS += [[side+'ShoulderTop',side+'ElbowOut',side+'WristOut',side+'HandOut'],
               ['Waist'+side+'Front',side+'ThighFront',side+'KneeOut',side+'Shin',side+'Heel',side+'ToeTip']]


class Demo:
    def __init__(self, root, recording):
        self.root, self.r = root, recording
        self.playing, self.index, self.last_tick = False, 0, time.monotonic()
        self.fullscreen = False
        self.window_seconds = tk.IntVar(value=5)
        self.emotion_cache = {}
        self.emotion_future = None
        self.emotion_result = None
        self.emotion_error = None
        self.score_targets = {}
        self.score_values = {}
        self.animation_time = time.monotonic()
        self.display_context = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        try:
            from emotion_model import EmotionModel, map_markers
            if not hasattr(self,'emotion_engine'):
                self.emotion_engine = EmotionModel()
            self.mapped_joints = map_markers(recording)
        except Exception as error:
            self.emotion_error = str(error)
        self.protective_cache = {}
        self.protective_future = None
        self.protective_result = None
        self.protective_error = None
        try:
            from protective_model import ProtectiveModel, map_markers as map_protective
            if not hasattr(self,'protective_engine'):
                self.protective_engine = ProtectiveModel()
            self.protective_joints = map_protective(recording)
        except Exception as error:
            self.protective_error = str(error)
        root.tk.call('tk', 'scaling', 1.25)
        root.title('Motion & Well-being | Research Showcase')
        root.geometry(f'{min(1440,root.winfo_screenwidth()-80)}x{min(950,root.winfo_screenheight()-100)}')
        root.minsize(1100, 800)
        root.configure(bg=BG)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Segoe UI', 10), padding=8,
                        background='#eaf0f8',foreground=FG,borderwidth=0)
        style.map('TButton',background=[('active','#dbe8f5')])
        style.configure('TCombobox',fieldbackground=BG,background=PANEL,foreground=FG)
        for name,color in COLORS.items():
            style.configure(name+'.Horizontal.TProgressbar',background=color,
                            troughcolor='#e2e8f1',bordercolor=PANEL,lightcolor=color,darkcolor=color)
        root.bind('<F11>', self.toggle_fullscreen)
        root.bind('<Escape>', lambda _: root.attributes('-fullscreen', False))
        root.bind('<space>', lambda _: self.toggle())
        header = tk.Frame(root, bg=BG)
        header.grid(row=0,column=0,sticky='ew',padx=26,pady=(18,5))
        ttk.Button(header,text='Open C3D…',command=self.open_recording).pack(side='right')
        tk.Label(header, text='MOTION  /  WELL-BEING', bg=BG, fg=CYAN,
                 font=('Segoe UI', 12, 'bold')).pack(anchor='w')
        tk.Label(header, text='Gait & movement', bg=BG, fg=FG,
                 font=('Segoe UI', 26, 'bold')).pack(anchor='w')
        tk.Label(header, text=f'RECORDED PLAYBACK   •   {recording.path.name}   •   {len(recording.time)/recording.rate:.1f} s   •   {recording.rate:g} Hz',
                 bg=BG, fg=MUTED, font=('Segoe UI', 10)).pack(anchor='w', pady=4)
        cards = tk.Frame(root, bg=BG)
        cards.grid(row=1,column=0,sticky='ew',padx=22,pady=8)
        self.card_values = []
        for title, subtitle in [('MEAN WALKING SPEED', 'Pelvis speed • selected rolling window'),
                                ('MEAN RELATIVE WRIST MOTION', 'Relative wrist speed • selected rolling window'),
                                ('ARM MOVEMENT RANGE', 'Left / right • selected rolling window')]:
            card = tk.Frame(cards, bg=PANEL, padx=18, pady=10)
            card.pack(side='left', expand=True, fill='both', padx=4)
            tk.Label(card, text=title, bg=PANEL, fg=MUTED, font=('Segoe UI', 10, 'bold')).pack(anchor='w')
            value = tk.Label(card, text='—', bg=PANEL, fg=[CYAN,'#6b55bd','#c05a36'][len(self.card_values)], font=('Segoe UI', 23, 'bold'))
            value.pack(anchor='w')
            self.card_values.append(value)
            tk.Label(card, text=subtitle, bg=PANEL, fg=MUTED, font=('Segoe UI', 9)).pack(anchor='w')

        main = tk.Frame(root, bg=BG)
        main.grid(row=2,column=0,sticky='nsew',padx=26)
        main.pack_propagate(False)
        chart_frame = tk.Frame(main, bg=BG)
        chart_frame.pack(side='left', expand=True, fill='both')
        self.fig = Figure(figsize=(10, 4.4), facecolor=BG)
        self.ax3 = self.fig.add_subplot(121, projection='3d', facecolor=BG)
        self.ax1 = self.fig.add_subplot(222, facecolor=BG)
        self.ax2 = self.fig.add_subplot(224, facecolor=BG)
        self.fig.subplots_adjust(left=.01, right=.98, bottom=.13, top=.9, wspace=.35, hspace=.5)
        for ax in [self.ax3,self.ax1,self.ax2]:
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.title.set_color(FG)
            for spine in ax.spines.values():
                spine.set_color(PANEL)
        self.ax3.set_title('Motion playback', fontsize=11)
        self.ax3.set(xlim=(-1.1,1.1),ylim=(-1.1,1.1),zlim=(0,2.1))
        self.ax3.set_box_aspect((1,1,1))
        self.ax3.view_init(elev=15, azim=-65)
        for axis in [self.ax3.xaxis,self.ax3.yaxis,self.ax3.zaxis]:
            axis.set_pane_color((.96,.97,.99,1))
        self.ax3.set_xlabel('X (m)', color=MUTED, fontsize=8)
        self.ax3.set_ylabel('Y (m)', color=MUTED, fontsize=8)
        self.ax3.set_zlabel('Height (m)', color=MUTED, fontsize=8)
        self.dots, = self.ax3.plot([],[],[], 'o', color=CYAN, markersize=4)
        self.bones = []
        for chain in CHAINS:
            indices = [recording.lookup[n] for n in chain if n in recording.lookup]
            line, = self.ax3.plot([],[],[], color=FG, lw=1.6, alpha=.8)
            self.bones.append((line, indices))
        stride = max(1, round(recording.rate / 30))
        self.ax1.plot(recording.time[::stride],recording.speed[::stride],color=CYAN,lw=1)
        self.ax1.set_title('Gait speed', fontsize=10)
        self.ax1.set_ylabel('m/s', color=MUTED)
        self.ax1.set_xlabel('Time (s)',color=MUTED)
        self.ax2.plot(recording.time[::stride],recording.trunk_lean[::stride],color=GOLD,lw=1)
        self.ax2.set_title('Trunk lean', fontsize=10)
        self.ax2.set_ylabel('Angle from upright (°)', color=MUTED,fontsize=8)
        self.ax2.set_xlabel('Time (s)',color=MUTED)
        self.ax2.set_ylim(bottom=0)
        self.cursors = []
        for ax in [self.ax1,self.ax2]:
            ax.set_xlim(0,recording.time[-1])
            ax.grid(alpha=.12)
            self.cursors.append(ax.axvline(0,color=MUTED,alpha=.65,lw=1))
        self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.canvas.get_tk_widget().pack(fill='both',expand=True)

        side = tk.Frame(main,bg=PANEL,padx=16,pady=16,width=300)
        side.pack(side='right',fill='y',padx=(12,0))
        self.label(side,'GAIT EXPRESSION',CYAN,11,True)
        self.label(side,'Model scores',FG,15,True)
        self.label(side,'ST-GCN · Take an Emotion Walk',MUTED,9)
        self.emotion_rows = {}
        for name in ['Happy','Sad','Angry','Neutral']:
            row = tk.Frame(side,bg=PANEL)
            row.pack(fill='x',pady=3)
            tk.Label(row,text=name,bg=PANEL,fg=FG,width=8,anchor='w').pack(side='left')
            bar = ttk.Progressbar(row,maximum=1,length=115,style=name+'.Horizontal.TProgressbar')
            bar.pack(side='left',padx=5)
            score = tk.Label(row,text='—',bg=PANEL,fg=COLORS[name],width=6)
            score.pack(side='left')
            self.emotion_rows[name] = (bar,score)
        self.emotion_status = tk.Label(side,text='Loading…',bg=PANEL,fg=GOLD,
                                      wraplength=265,justify='left',height=3,font=('Segoe UI',9))
        self.emotion_status.pack(anchor='w',pady=10)
        self.label(side,'PROTECTIVE MOVEMENT',COLORS['Protective'],11,True)
        self.label(side,'EmoPain-Fusion · pose branch',MUTED,9)
        self.protective_rows = {}
        for name in ['Non-protective','Protective']:
            row = tk.Frame(side,bg=PANEL)
            row.pack(fill='x',pady=3)
            tk.Label(row,text=name,bg=PANEL,fg=FG,width=13,anchor='w').pack(side='left')
            bar = ttk.Progressbar(row,maximum=1,length=80,style=name+'.Horizontal.TProgressbar')
            bar.pack(side='left',padx=5)
            score = tk.Label(row,text='—',bg=PANEL,fg=COLORS[name],width=6)
            score.pack(side='left')
            self.protective_rows[name] = (bar,score)
        self.protective_status = tk.Label(side,text='Loading…',bg=PANEL,fg=GOLD,
                                        wraplength=265,justify='left',height=3,font=('Segoe UI',9))
        self.protective_status.pack(anchor='w',pady=6)

        controls = tk.Frame(root,bg=BG)
        controls.grid(row=3,column=0,sticky='ew',padx=26,pady=5)
        window_controls = tk.Frame(controls,bg=BG)
        window_controls.pack(side='top',fill='x',pady=(0,5))
        tk.Label(window_controls,text='Rolling window',bg=BG,fg=FG).pack(side='left')
        selector = ttk.Combobox(window_controls,textvariable=self.window_seconds,
                                values=list(range(3,16)),state='readonly',width=4)
        selector.pack(side='left',padx=6)
        selector.bind('<<ComboboxSelected>>',lambda _: self.render())
        tk.Label(window_controls,text='seconds',bg=BG,fg=MUTED).pack(side='left')
        self.window_label = tk.Label(window_controls,text='',bg=BG,fg=CYAN)
        self.window_label.pack(side='left',padx=15)
        self.play_button = ttk.Button(controls,text='Play',command=self.toggle)
        self.play_button.pack(side='left')
        ttk.Button(controls,text='Restart',command=self.restart).pack(side='left',padx=6)
        self.position = tk.DoubleVar(value=0)
        self.seek = tk.Scale(controls,variable=self.position,from_=0,to=recording.time[-1],resolution=.01,
                             orient='horizontal',showvalue=False,bg=BG,fg=FG,troughcolor=PANEL,
                             highlightthickness=0,command=self.on_seek)
        self.seek.pack(side='left',fill='x',expand=True,padx=8)
        self.clock_label = tk.Label(controls,text='',bg=BG,fg=FG,width=17)
        self.clock_label.pack(side='left')
        ttk.Button(controls,text='Export',command=self.export).pack(side='right')
        ttk.Button(controls,text='Full screen',command=self.toggle_fullscreen).pack(side='right',padx=6)

        self.quality = tk.Label(root,text='',bg=BG,fg=MUTED,font=('Segoe UI',9))
        self.quality.grid(row=4,column=0,sticky='w',padx=26,pady=(3,10))
        self.render()
        self.timer = root.after(33,self.tick)

    def open_recording(self):
        path = filedialog.askopenfilename(title='Open motion recording',
            filetypes=[('C3D motion capture','*.c3d')],initialdir=self.r.path.parent)
        if path:
            self.load_recording(path)

    def load_recording(self,path):
        try:
            recording = Recording(path)
        except Exception as error:
            messagebox.showerror('Cannot open recording',f'{error}\n\nUse a single-subject C3D with the same full-body marker set as the sample.')
            return False
        seconds = self.window_seconds.get()
        self.root.after_cancel(self.timer)
        self.executor.shutdown(wait=False,cancel_futures=True)
        self.playing = False
        for child in self.root.winfo_children():
            child.destroy()
        self.fig.clear()
        self.__init__(self.root,recording)
        self.window_seconds.set(seconds)
        self.render()
        return True

    def label(self,parent,text,color,size=10,bold=False,wrap=265):
        tk.Label(parent,text=text,bg=parent.cget('bg'),fg=color,font=('Segoe UI',size,'bold' if bold else 'normal'),
                 wraplength=wrap,justify='left').pack(anchor='w',pady=(0,6))

    def toggle_fullscreen(self,event=None):
        self.root.attributes('-fullscreen',not bool(self.root.attributes('-fullscreen')))

    def toggle(self):
        self.playing = not self.playing
        self.last_tick = time.monotonic()
        self.play_button.configure(text='Pause' if self.playing else 'Play')

    def restart(self):
        self.position.set(0)
        self.index = 0
        self.render()

    def on_seek(self,value):
        self.index = min(len(self.r.time)-1,round(float(value)*self.r.rate))
        self.render()

    def tick(self):
        now = time.monotonic()
        if self.playing:
            t = (self.position.get()+now-self.last_tick) % (self.r.time[-1]+1/self.r.rate)
            self.position.set(t)
            self.index = min(len(self.r.time)-1,round(t*self.r.rate))
            self.render()
        self.last_tick = now
        self.update_emotion()
        self.animate_scores()
        self.timer = self.root.after(33,self.tick)

    def update_emotion(self):
        self.update_protective()
        if self.emotion_error:
            self.emotion_status.configure(text=f'Model unavailable: {self.emotion_error}')
            return
        seconds = self.window_seconds.get()
        # Inference at 2 Hz during playback; exact frame when paused/scrubbing.
        step = max(1,round(self.r.rate/2))
        frame = self.index//step*step if self.playing else self.index
        key = (frame,seconds)
        if self.emotion_future is not None and self.emotion_future.done():
            try:
                self.emotion_cache[self.pending_key] = self.emotion_future.result()
            except Exception as error:
                self.emotion_cache[self.pending_key] = {'scores':None,'status':f'Inference failed: {error}'}
            self.emotion_future = None
        result = self.emotion_cache.get(key)
        self.emotion_result = result
        if result is None and self.emotion_future is None:
            self.pending_key = key
            self.emotion_future = self.executor.submit(self.emotion_engine.predict,
                self.r,self.mapped_joints,frame,seconds)
        if result is None and self.playing:
            recent = [k for k in self.emotion_cache if k[1] == seconds and 0 <= frame-k[0] <= self.r.rate]
            if recent:
                result = self.emotion_cache[max(recent)]
        if result is None:
            self.emotion_status.configure(text='Computing selected window…')
            if self.emotion_future is None:
                self.pending_key = key
                self.emotion_future = self.executor.submit(self.emotion_engine.predict,
                    self.r,self.mapped_joints,frame,seconds)
            return
        scores = result['scores']
        self.set_score_targets('emotion',scores)
        interval = f"{result.get('start_s',0):.1f}–{result.get('end_s',0):.1f} s\n"
        self.emotion_status.configure(text=interval if scores else interval+result['status'])

    def update_protective(self):
        if self.protective_error:
            self.protective_status.configure(text=f'Model unavailable: {self.protective_error}')
            return
        seconds = self.window_seconds.get()
        step = max(1,round(self.r.rate/2))
        frame = self.index//step*step if self.playing else self.index
        key = (frame,seconds)
        if self.protective_future is not None and self.protective_future.done():
            try:
                self.protective_cache[self.protective_pending_key] = self.protective_future.result()
            except Exception as error:
                self.protective_cache[self.protective_pending_key] = {'scores':None,'status':f'Inference failed: {error}'}
            self.protective_future = None
        result = self.protective_cache.get(key)
        self.protective_result = result
        if result is None and self.protective_future is None:
            self.protective_pending_key = key
            self.protective_future = self.executor.submit(self.protective_engine.predict,
                self.r,self.protective_joints,frame,seconds)
        if result is None and self.playing:
            recent = [k for k in self.protective_cache if k[1] == seconds and 0 <= frame-k[0] <= self.r.rate]
            if recent:
                result = self.protective_cache[max(recent)]
        scores = result.get('scores') if result else None
        if result is not None:
            self.set_score_targets('protective',scores)
        if result is None:
            self.protective_status.configure(text='Computing selected window…')
            if self.protective_future is None:
                self.protective_pending_key = key
                self.protective_future = self.executor.submit(self.protective_engine.predict,
                    self.r,self.protective_joints,frame,seconds)
        else:
            interval = f"{result.get('start_s',0):.1f}–{result.get('end_s',0):.1f} s"
            self.protective_status.configure(text=interval if scores else result['status'])

    def set_score_targets(self,kind,scores):
        rows = getattr(self,kind+'_rows')
        for name,(bar,label) in rows.items():
            if scores is None:
                self.score_targets.pop(name,None)
                self.score_values.pop(name,None)
                bar['value'] = 0
                label.configure(text='\u2014')
            else:
                self.score_targets[name] = float(scores[name])
                self.score_values.setdefault(name,0.)

    def animate_scores(self,now=None):
        now = time.monotonic() if now is None else now
        dt = max(0.,now-self.animation_time)
        self.animation_time = now
        # Time-based easing: ~95% of the transition in 0.75 seconds.
        alpha = 1-np.exp(-dt/.25)
        rows = {**self.emotion_rows,**self.protective_rows}
        for name,target in self.score_targets.items():
            value = self.score_values[name]+alpha*(target-self.score_values[name])
            if abs(target-value)<.0005:
                value = target
            self.score_values[name] = value
            bar,label = rows[name]
            bar['value'] = value
            label.configure(text=f'{value:.3f}')

    @staticmethod
    def fmt(value):
        return f'{value:.2f}' if np.isfinite(value) else '—'

    def render(self):
        r, i = self.r, self.index
        context = (i,self.window_seconds.get())
        if self.display_context is not None:
            previous,seconds = self.display_context
            if seconds != context[1] or i < previous or abs(i-previous)>r.rate:
                self.set_score_targets('emotion',None)
                self.set_score_targets('protective',None)
        self.display_context = context
        xyz = r.xyz[i].copy()
        xyz[:,:2] -= r.pelvis[i,:2]
        self.dots.set_data_3d(*xyz.T)
        for line, indices in self.bones:
            line.set_data_3d(*xyz[indices].T)
        t = r.time[i]
        seconds = self.window_seconds.get()
        start,end,s = r.rolling_summary(i,seconds)
        for cursor in self.cursors:
            cursor.set_xdata([t,t])
        for ax in [self.ax1,self.ax2]:
            ax.set_xlim(start,max(float(seconds),end))
        self.card_values[0].configure(text=f"{self.fmt(s['Walking speed (m/s)'])} m/s")
        self.card_values[1].configure(text=f"{self.fmt(s['Relative wrist speed (m/s)'])} m/s")
        value = f"{self.fmt(s['Left arm range (deg)'])}° / {self.fmt(s['Right arm range (deg)'])}°"
        self.card_values[2].configure(text=value)
        partial = ' • filling window' if end < seconds else ''
        self.window_label.configure(text=f'{start:.1f}–{end:.1f} s{partial}')
        self.clock_label.configure(text=f'{t:05.1f} / {r.time[-1]:05.1f} s')
        self.quality.configure(text=f'Markers: {r.valid[i].sum()}/{len(r.labels)}  •  Space: play/pause · F11: full screen')
        self.canvas.draw_idle()
        self.update_emotion()

    def export(self):
        folder = Path(__file__).parent/'outputs'
        self.r.export(folder)
        start,end,summary = self.r.rolling_summary(self.index,self.window_seconds.get())
        with (folder/'rolling_window.csv').open('w',newline='',encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['recording','window_seconds','start_s','end_s',*summary])
            writer.writerow([self.r.path.name,self.window_seconds.get(),start,end,*summary.values()])
        self.fig.savefig(folder/'showcase.png',dpi=160,facecolor=BG)
        if self.emotion_result is not None:
            (folder/'model_prediction.json').write_text(json.dumps({
                'recording':self.r.path.name,'model':'TAEW release ST-GCN baseline',
                'interpretation':'Experimental perceived-expression scores; not calibrated mental-state probabilities.',
                **self.emotion_result},indent=2),encoding='utf-8')
        else:
            (folder/'model_prediction.json').write_text(json.dumps({
                'recording':self.r.path.name,'scores':None,
                'status':self.emotion_error or 'Current prediction pending; wait and export again.'},indent=2),encoding='utf-8')
        (folder/'protective_prediction.json').write_text(json.dumps({
            'recording':self.r.path.name,
            'interpretation':'Unvalidated cross-sensor transfer of the pose subhead, not the full pose+EMG ensemble or a pain intensity estimate.',
            **(self.protective_result or {'scores':None,'status':self.protective_error or 'Prediction pending.'})
        },indent=2),encoding='utf-8')
        messagebox.showinfo('Export complete',f'Measurements, model scores and figure saved to:\n{folder}')


def main():
    # Avoid Windows bitmap scaling; let the layout use physical screen dimensions.
    import sys
    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
    parser = argparse.ArgumentParser()
    parser.add_argument('recording',nargs='?')
    parser.add_argument('--snapshot',action='store_true',help='Render a preview and exit (smoke test).')
    args = parser.parse_args()
    path = Path(args.recording) if args.recording else next(Path(__file__).parent.glob('*.c3d'),None)
    root = tk.Tk()
    if path is None:
        root.withdraw()
        selected = filedialog.askopenfilename(title='Open motion recording',
            filetypes=[('C3D motion capture','*.c3d')])
        if not selected:
            root.destroy()
            return
        path = Path(selected)
        root.deiconify()
    try:
        r = Recording(path)
    except Exception as error:
        messagebox.showerror('Cannot open recording',str(error))
        root.destroy()
        return
    app = Demo(root,r)
    if args.snapshot:
        root.update()
        app.index = min(round(15*r.rate),len(r.time)-1)
        app.render()
        root.update()
        folder = Path(__file__).parent/'outputs'
        r.export(folder)
        app.fig.savefig(folder/'showcase.png',dpi=140,facecolor=BG)
        print('Snapshot and measurements exported.',r.summary(0,r.time[-1]))
        root.destroy()
    else:
        root.mainloop()


if __name__ == '__main__':
    main()
