"""LLM counterpart of the CV normalization diagram: X[B, S, H]."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, FancyBboxPatch
from matplotlib.font_manager import FontProperties

font = FontProperties(fname='/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
fig,ax=plt.subplots(figsize=(15,7.1),dpi=170)
ax.set(xlim=(0,15),ylim=(0,7.1),aspect='equal');ax.axis('off')
fig.patch.set_facecolor('white')
INK='#263443';GRID='#8993a0'
def text(x,y,t,size=17,color=INK,ha='center',weight='normal'):
    ax.text(x,y,t,fontsize=size,fontproperties=font,color=color,ha=ha,va='center',weight=weight,zorder=50)
def face(points,color,edge=GRID,lw=1):
    ax.add_patch(Polygon(points,closed=True,facecolor=color,edgecolor=edge,linewidth=lw))
def arrow(a,b,c=INK,lw=1.5):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,color=c,linewidth=lw))

def cube(ox,oy,c,light,title):
    # x=batch, vertical=hidden, depth=sequence. First token vector is visible
    # on the front face: four vertically stacked cells for b=0, s=0.
    w,h,dx,dy=.96,.52,-.35,.31
    def p(b,s,j):return (ox+b*w+s*dx,oy+j*h+s*dy)
    for s in [2,1,0]:
        # Left side spans sequence positions; the front band is s=0.
        for j in range(4):
            face([p(0,s,j),p(0,s+1,j),p(0,s+1,j+1),p(0,s,j+1)],c if s==0 else '#f1f3f5')
        for b in range(2):
            face([p(b,s,4),p(b+1,s,4),p(b+1,s+1,4),p(b,s+1,4)],light if b==0 and s==0 else '#f7f8fa')
    for j in range(4):
        for b in range(2):
            face([p(b,0,j),p(b+1,0,j),p(b+1,0,j+1),p(b,0,j+1)],light if b==0 else '#ffffff')
            if b==0:text(ox+w*.5,oy+(j+.5)*h,f'x{3-j}',16,color=c,weight='bold')
    text(ox+.35,6.7,title,25,weight='bold')
    text(ox+.35,6.15,'X 的形状：[B, S, H] = [2, 3, 4]',15)
    # Independent arrows outside cube, faithful to original cuboid layout.
    origin=(ox-1.35,oy-.35)
    arrow(origin,(ox+2*w+.3,oy-.35))
    text(ox+.62,oy-.72,'Batch B：2 条文本',16)
    arrow(origin,(origin[0],oy+4*h+.72))
    text(origin[0]-.22,oy+2.10,'Hidden H',16,ha='right')
    text(origin[0]-.22,oy+1.69,'4 个特征',14,ha='right')
    arrow(origin,(origin[0]-1.05,oy+.6))
    text(origin[0]-.78,oy+1.03,'Token S',15)
    text(origin[0]-.81,oy+.62,'3 个位置',13)
    # callout for the selected b=0,s=0 fiber
    anchor=(ox+.5*w,oy+1.6*h)
    label=(ox+2.60,oy+2.36)
    arrow(label,anchor,c,1.4)
    text(label[0]+.28,label[1]+.36,'一个 token 的',14,color=c)
    text(label[0]+.28,label[1]+.02,'完整 hidden 向量',14,color=c)
    text(ox+.35,oy-.06,'b=0',12,color=c)
    text(ox+1.46,oy-.06,'b=1',12,color='#64748b')

cube(3.28,2.79,'#355dd4','#d4defa','LayerNorm')
cube(10.78,2.79,'#15846d','#c7eae0','RMSNorm')
text(3.63,1.53,'减均值 → 除标准差 → γ、β',18,color='#355dd4')
text(11.13,1.53,'除均方根 → γ（不减均值）',18,color='#15846d')
text(3.63,1.09,'均值 / 方差沿 H 维计算',16)
text(11.13,1.09,'平方均值沿 H 维计算',16)
ax.add_patch(FancyBboxPatch((.45,.12),14.1,.53,boxstyle='round,pad=.03,rounding_size=.05',facecolor='#f2f4f7',edgecolor='none'))
text(7.5,.385,'着色部分 = 一次计算统计量的范围：X[b, s, :]。其他 batch、token 各自独立计算。',16)
fig.savefig(Path(__file__).with_name('llm_normalization.png'),bbox_inches='tight',pad_inches=.05)
plt.close(fig)
