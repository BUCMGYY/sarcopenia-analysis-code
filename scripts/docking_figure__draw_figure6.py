"""Source-based Figure 6; no historic substituted docking systems are imported.

Run with --draft only for internal layout inspection while corrected runs finish.
The default final export refuses missing structures or invalid stereochemical rows.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image
import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle, FancyArrowPatch

HERE = Path(__file__).resolve().parent
config_path = Path(os.environ.get('FIGURE6_CONFIG', HERE / 'figure_config.json'))
if not config_path.is_absolute():
    config_path = (HERE / config_path).resolve()
if not config_path.exists():
    config_path = HERE / 'figure_config.example.json'
CFG = json.loads(config_path.read_text(encoding='utf-8'))
MM = 72 / 25.4
NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)
ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': [CFG['font'], 'DejaVu Sans'],
    'font.size': CFG['body_pt'], 'text.color': CFG['ink'],
    'axes.labelcolor': CFG['ink'], 'xtick.color': CFG['ink'], 'ytick.color': CFG['ink'],
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .7,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'legend.frameon': False,
})


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def axes_mm(fig, x, y, w, h):
    W, H = CFG['width_mm'], CFG['height_mm']
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])


def text_mm(fig, x, y, s, **kwargs):
    return fig.text(x / CFG['width_mm'], 1 - y / CFG['height_mm'], s,
                    va='baseline', ha='left', color=CFG['ink'], **kwargs)


def title(fig, x, y, letter, wording):
    # One baseline and fixed physical spacing, rather than axes-relative labels.
    text_mm(fig, x, y, letter, fontsize=CFG['title_pt'], fontweight='bold')
    text_mm(fig, x + 4.0, y, wording, fontsize=CFG['title_pt'], fontweight='bold')


def read_inputs(draft=False):
    dock = (HERE / CFG['docking_root']).resolve()
    candidates = pd.read_csv(HERE / CFG['drugreflector_source'])
    assert len(candidates) == 100 and candidates.compound_id.nunique() == 100
    assert np.array_equal(candidates.rank_1based.to_numpy(), np.arange(1, 101))
    assert np.isfinite(candidates['prob']).all()
    fixed = dock / 'input_repair_v2a' / 'figure_sources' / 'primary_state_score_summary.csv'
    score_path = fixed if fixed.exists() else dock / 'analysis_v2' / 'primary_state_score_summary.csv'
    scores = pd.read_csv(score_path)
    assert len(scores) == 12 and not scores.duplicated(['state_id', 'target']).any()
    scores = scores.set_index(['state_id', 'target']).reindex(
        pd.MultiIndex.from_product([CFG['state_order'], CFG['target_order']], names=['state_id', 'target'])
    ).reset_index()
    assets = json.loads((HERE / CFG['asset_manifest']).read_text(encoding='utf-8'))
    if not draft:
        assert len(assets) == 4, 'Refusing final export without all four real molecular state assets'
        assert {a['tag'] for a in assets} == set(CFG['structure_order'])
        assert scores.median_seed_top_vina_score.notna().all(), 'Refusing invalid/missing primary scores'
        assert scores.n_admissible_seed_scores.eq(3).all(), 'Each state must have three admissible seeds'
        assert fixed.exists(), 'AACOCF3 requires the independently repaired torsion-tree run'
    asset_map = {a['tag']: a for a in assets}
    for a in assets:
        assert sha(a['pose_path']) == a['pose_sha256'], 'Molecular source changed since rendering'
    return candidates, scores, asset_map, score_path


def draw_rank(fig, dr):
    title(fig, 7, 7, 'a', 'DrugReflector ranking')
    ax = axes_mm(fig, 19, 17, 63, 40)
    ax.plot(dr.rank_1based, dr.prob, color=CFG['pale_blue'], lw=1.0, zorder=1)
    ax.scatter(dr.rank_1based, dr.prob, s=8, color=CFG['blue'], linewidths=0, zorder=2)
    ax.scatter(dr.rank_1based.iloc[:3], dr.prob.iloc[:3], s=26, color=CFG['red'],
               edgecolors='white', linewidths=.4, zorder=4)
    label_y = [.0405, .0320, .0242]
    for (_, r), name, yy in zip(dr.head(3).iterrows(), ['JW-74', 'Manidipine', 'AACOCF3'], label_y):
        ax.annotate(f"{int(r.rank_1based)}  {name}", (r.rank_1based, r.prob),
                    xytext=(21, yy), textcoords='data', ha='left', va='center', fontsize=8,
                    color=CFG['ink'], arrowprops=dict(arrowstyle='-', lw=.65, color=CFG['red'],
                    shrinkA=2, shrinkB=3, connectionstyle='angle,angleA=0,angleB=90,rad=2'))
    ax.set(xlim=(-2, 102), ylim=(0, .045), xticks=[1, 25, 50, 75, 100], yticks=[0, .01, .02, .03, .04])
    ax.set_xlabel('Candidate rank', labelpad=4)
    ax.set_ylabel('Model probability', labelpad=5)
    ax.tick_params(labelsize=7, length=3, pad=3)
    ax.grid(axis='y', color=CFG['grid'], lw=.5)
    ax.set_axisbelow(True)


def draw_scores(fig, scores):
    title(fig, 94, 7, 'b', 'Compound–target docking')
    ax = axes_mm(fig, 121, 17, 52, 37)
    cmap = LinearSegmentedColormap.from_list('red_white_blue_scores',
        [CFG['red'], '#F4DDDE', '#FAFAFA', '#D4E2ED', CFG['blue']])
    norm = Normalize(*CFG['score_limits'])
    mat = scores.pivot(index='state_id', columns='target', values='median_seed_top_vina_score').reindex(
        index=CFG['state_order'], columns=CFG['target_order'])
    im = ax.imshow(np.ma.masked_invalid(mat.to_numpy(float)), cmap=cmap, norm=norm, aspect='auto')
    ax.set_xticks(range(3), ['TNKS2', 'CACNA1C', 'PLA2G4A'], fontsize=7)
    ax.set_yticks(range(4), CFG['state_labels'], fontsize=7)
    ax.tick_params(length=0, pad=4)
    for sp in ax.spines.values(): sp.set_visible(False)
    for i in range(4):
        for j in range(3):
            v = mat.iloc[i, j]
            if pd.isna(v):
                ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, facecolor='white', edgecolor=CFG['grid'], hatch='///', lw=.4))
                label = '—'
            else: label = f'{v:.2f}'
            ax.text(j, i, label, ha='center', va='center', color=CFG['ink'], fontsize=8.1)
    ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.4)
    ax.tick_params(which='minor', bottom=False, left=False)
    ca = axes_mm(fig, 125, 63, 44, 2.2)
    cb = fig.colorbar(im, cax=ca, orientation='horizontal', ticks=[-12, -10, -8, -6])
    cb.ax.tick_params(labelsize=7, length=2, pad=2)
    cb.outline.set_linewidth(.5)
    fig.text(147/CFG['width_mm'],1-14/CFG['height_mm'],
             'Median Vina score (kcal/mol)',ha='center',va='baseline',fontsize=7)
    return mat


def trim_bounds(path):
    im = Image.open(path).convert('RGB')
    a = np.asarray(im)
    yy, xx = np.where((a < 246).any(axis=2))
    if not len(xx): raise ValueError(f'Empty molecular image: {path.name}')
    pad = 8
    return (max(0, int(xx.min())-pad), max(0, int(yy.min())-pad),
            min(im.width, int(xx.max())+pad+1), min(im.height, int(yy.max())+pad+1)), im.size


def fit_rect(source_w, source_h, rect):
    x, y, w, h = rect
    scale = min(w/source_w, h/source_h)
    fw, fh = source_w*scale, source_h*scale
    return [x+(w-fw)/2, y+(h-fh)/2, fw, fh]


def add_image(fig, path, rect, key, trim=True):
    im = Image.open(path).convert('RGB')
    crop, orig = trim_bounds(path) if trim else ((0,0,*im.size), im.size)
    im = im.crop(crop)
    fitted = fit_rect(*im.size, rect)
    ax = axes_mm(fig, *fitted)
    ax.set_gid(key)
    ax.imshow(im, interpolation='none', aspect='equal')
    ax.set_axis_off()
    box=None
    if key.startswith('overview_'):
        rgb=np.asarray(im,dtype=float)
        # The receptor is blue and the ligand uniquely red in these source renders.
        red=(rgb[:,:,0]>1.35*rgb[:,:,1]) & (rgb[:,:,0]>1.2*rgb[:,:,2]) & (rgb[:,:,0]>90)
        ry,rx=np.where(red)
        if len(rx):
            x0,y0,x1,y1=max(0,rx.min()-4),max(0,ry.min()-4),min(im.width,rx.max()+5),min(im.height,ry.max()+5)
            box=[int(x0),int(y0),int(x1),int(y1)]
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=CFG['blue'],lw=.55))
    return dict(key=key, path=str(path), crop=crop, original_size=orig, fitted_rect_mm=fitted,
                overview_ligand_box_px=box,
                aspect_error=abs((fitted[2]/fitted[3])/(im.width/im.height)-1))


def zoom_arrow(fig, x0, x1, y, key):
    arrow=FancyArrowPatch(
        (x0/CFG['width_mm'],1-y/CFG['height_mm']),
        (x1/CFG['width_mm'],1-y/CFG['height_mm']),
        arrowstyle='-|>', mutation_scale=CFG['zoom_arrow_head_scale_pt'],
        linewidth=CFG['zoom_arrow_linewidth_pt'], color=CFG['blue'],
        shrinkA=0, shrinkB=0, transform=fig.transFigure, clip_on=False,
        zorder=10)
    arrow.set_gid(key)
    fig.add_artist(arrow)


def visible_edge_at_height(placement, y, side):
    im=Image.open(placement['path']).convert('RGB').crop(placement['crop'])
    pixels=np.asarray(im)
    x0,y0,w,h=placement['fitted_rect_mm']
    band=CFG['zoom_arrow_contour_band_half_height_mm']
    lo=max(0,int(np.floor((y-band-y0)/h*im.height)))
    hi=min(im.height,int(np.ceil((y+band-y0)/h*im.height)))
    assert hi>lo
    _,xx=np.where((pixels[lo:hi]<246).any(axis=2))
    assert len(xx),'Connector height must intersect visible molecular content'
    edge=float(xx.max()+1) if side=='right' else float(xx.min())
    return x0+edge/im.width*w


def centered_zoom_arrow(fig, overview, detail, y, key, length_mm):
    left=visible_edge_at_height(overview,y,'right')
    right=visible_edge_at_height(detail,y,'left')
    assert right-left>length_mm, 'Arrow must fit inside the actual inter-image gap'
    center=(left+right)/2
    x0=center-length_mm/2
    x1=center+length_mm/2
    zoom_arrow(fig,x0,x1,y,key)
    return dict(key=key,small_visible_right_mm=left,large_visible_left_mm=right,
                edge_rule='nonwhite source pixels in +/-1 mm band at connector height',
                arrow_start_mm=x0,arrow_end_mm=x1,arrow_y_mm=y,
                arrow_center_mm=center,left_clearance_mm=x0-left,
                right_clearance_mm=right-x1)


def draw_structures(fig, assets, draft):
    placements=[]
    arrows=[]
    directory = HERE / 'molecular_assets'
    singles=[(CFG['structure_order'][0], CFG['row_title_y_mm'][0], 'c'),
             (CFG['structure_order'][-1], CFG['row_title_y_mm'][-1], 'e')]
    for tag, yy, letter in singles:
        if tag not in assets:
            if not draft: raise ValueError(f'Missing valid asset {tag}')
            continue
        a=assets[tag]
        title(fig, 7, yy, letter, f"{a['compound']}–{a['target']} ({a['pdb']})")
        placements.append(add_image(fig, directory/f'{tag}_overview.png', [7, yy+14, 20, 24], f'overview_{letter}'))
        placements.append(add_image(fig, directory/f'{tag}_detail.png', [29, yy+5, 58, 44], f'detail_{letter}'))
        placements.append(add_image(fig, directory/f'{tag}_contacts2d.png', [92, yy+5, 84, 44], f'contact_{letter}'))
        arrows.append(centered_zoom_arrow(fig, placements[-3], placements[-2],
                      yy+26, f'zoom_arrow_{letter}', 2.0))
    yy=CFG['row_title_y_mm'][1]
    title(fig, 7, yy, 'd', 'Manidipine–CACNA1C (8HMB)')
    for state, x, isomer in zip(CFG['structure_order'][1:3], CFG['isomer_column_x_mm'], ['R','S']):
        if state not in assets:
            if not draft: raise ValueError(f'Missing valid asset {state}')
            continue
        fig.text((x+41)/CFG['width_mm'], 1-(yy+CFG['isomer_subtitle_offset_mm'])/CFG['height_mm'],
                 isomer, ha='center', va='baseline', color=CFG['ink'],
                 fontsize=8.5, fontweight='bold')
        tag=state
        placements.append(add_image(fig, directory/f'{tag}_overview.png',
                          [x, yy+20, 17, 20], f'overview_d_{isomer}'))
        placements.append(add_image(fig, directory/f'{tag}_detail.png',
                          [x+20, yy+CFG['isomer_detail_offset_mm'], 62, 35], f'detail_d_{isomer}'))
        placements.append(add_image(fig, directory/f'{tag}_contacts2d.png',
                          [x, yy+CFG['isomer_contact_offset_mm'], 82, 43], f'contact_d_{isomer}'))
        arrows.append(centered_zoom_arrow(fig, placements[-3], placements[-2],
                      yy+30, f'zoom_arrow_d_{isomer}', 3.0))
    return placements,arrows


def overlay_vectors(pdf_path, svg_path, placements):
    """Replace 2D raster previews with corresponding native vector assets."""
    doc=fitz.open(pdf_path)
    root=ET.parse(svg_path)
    for p in placements:
        if not p['key'].startswith('contact_'): continue
        png=Path(p['path']); vec=png.with_name(png.stem+'_vector.svg')
        vecpdf=png.with_name(png.stem+'_vector.pdf')
        if not vec.exists() or not vecpdf.exists():
            continue
        x,y,w,h=p['fitted_rect_mm']; crop=p['crop']; ow,oh=p['original_size']
        srcpdf=fitz.open(vecpdf); page=srcpdf[0]
        clip=fitz.Rect(crop[0]/ow*page.rect.width,crop[1]/oh*page.rect.height,
                       crop[2]/ow*page.rect.width,crop[3]/oh*page.rect.height)
        dest=fitz.Rect(x*MM,y*MM,(x+w)*MM,(y+h)*MM)
        doc[0].draw_rect(dest, color=None, fill=(1,1,1), overlay=True)
        doc[0].show_pdf_page(dest,srcpdf,0,clip=clip,keep_proportion=True,overlay=True)
        srcpdf.close()
        # Main SVG gets a nested SVG with the same crop and isotropic fitting.
        parent_map={c:pr for pr in root.iter() for c in pr}
        for node in list(root.iter()):
            if node.get('id')==p['key']:
                parent_map[node].remove(node)
        source=ET.parse(vec).getroot(); vb=[float(z)for z in source.get('viewBox').split()]
        # The final figure already has a white background. Remove only the
        # two full-canvas white rectangles (outer source and RDKit canvas).
        # Some SVG review engines do not honor nested clipPaths and would
        # otherwise let these rectangles obscure neighbouring panels.
        # Small white residue-label guards remain intact.
        parents={child:parent for parent in source.iter() for child in parent}
        for node in list(source.iter()):
            if node.tag == '{%s}rect'%NS:
                width=float(node.get('width','0').removesuffix('px'))
                height=float(node.get('height','0').removesuffix('px'))
                white=node.get('fill','') in ('white','#FFFFFF','#ffffff') or 'fill:#FFFFFF' in node.get('style','')
                if white and width>=700 and height>=500:
                    parents[node].remove(node)
        vx=vb[0]+crop[0]/ow*vb[2];vy=vb[1]+crop[1]/oh*vb[3]
        vw=(crop[2]-crop[0])/ow*vb[2];vh=(crop[3]-crop[1])/oh*vb[3]
        # Explicit group transforms avoid nested-SVG viewport incompatibilities
        # in PDF/SVG review engines while retaining the very same geometry.
        defs=ET.SubElement(root.getroot(),'{%s}defs'%NS)
        clipid=p['key']+'_global_clip'
        cp=ET.SubElement(defs,'{%s}clipPath'%NS,{'id':clipid,'clipPathUnits':'userSpaceOnUse'})
        # Text renderers have slightly different glyph bearings; a 2-mm
        # outer guard prevents clipping editable contact labels. It changes
        # neither the chemical geometry nor the fitted image aspect ratio.
        guard=2.0
        ET.SubElement(cp,'{%s}rect'%NS,dict(x=str((x-guard)*MM),y=str((y-guard)*MM),
                     width=str((w+2*guard)*MM),height=str((h+2*guard)*MM)))
        outer=ET.Element('{%s}g'%NS,{'clip-path':f'url(#{clipid})'})
        wrap=ET.SubElement(outer,'{%s}g'%NS,{'transform':
            f'translate({x*MM} {y*MM}) scale({w*MM/vw} {h*MM/vh}) translate({-vx} {-vy})'})
        # Namespace IDs to avoid collisions with other matplotlib SVG panels.
        serial=p['key']+'_'
        mapping={n.get('id'):serial+n.get('id')for n in source.iter()if n.get('id')}
        for n in source.iter():
            for k,v in list(n.attrib.items()):
                if k=='id': n.set(k,mapping[v]);continue
                for old,new in mapping.items():
                    v=v.replace(f'url(#{old})',f'url(#{new})')
                    if v=='#'+old:v='#'+new
                n.set(k,v)
        for child in list(source):
            copied=copy.deepcopy(child)
            if copied.tag=='{%s}svg'%NS:
                # RDKit's child has a 0 0 1000 800 viewBox and matching dimensions.
                cv=[float(z)for z in copied.get('viewBox','0 0 1000 800').split()]
                assert cv==[0.,0.,1000.,800.]
                g=ET.SubElement(wrap,'{%s}g'%NS)
                for element in list(copied):g.append(element)
            else:wrap.append(copied)
        root.getroot().append(outer)
    tmp=pdf_path.with_name(pdf_path.stem+'_composed.pdf')
    doc.save(tmp,garbage=4,deflate=True);doc.close()
    tmp.replace(pdf_path)
    # Expand Matplotlib's CSS font shorthand for portable editable SVG text.
    for node in root.iter():
        style=node.get('style','')
        match=re.search(r'font:\s*([^;]+)',style)
        if match:
            value=match.group(1)
            size=re.search(r'([0-9.]+)px',value)
            if size:
                node.set('font-size',size.group(1))
                node.set('font-family','Arial')
                node.set('font-weight','bold' if 'bold' in value or '700' in value else 'normal')
                style=re.sub(r'font:\s*[^;]+;?','',style)
                node.set('style',style)
    root.write(svg_path,encoding='utf-8',xml_declaration=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--draft',action='store_true');args=ap.parse_args()
    dr,scores,assets,score_path=read_inputs(args.draft)
    fig=plt.figure(figsize=(CFG['width_mm']/25.4,CFG['height_mm']/25.4))
    draw_rank(fig,dr);mat=draw_scores(fig,scores);placements,arrows=draw_structures(fig,assets,args.draft)
    basename='Figure6_layout_review' if args.draft else 'Figure6_original_top3_drug_docking'
    pdf=HERE/(basename+'.pdf');svg=HERE/(basename+'.svg')
    fig.savefig(pdf);fig.savefig(svg)
    plt.close(fig)
    overlay_vectors(pdf,svg,placements)
    document=fitz.open(pdf)
    for dpi,suffix in [(170 if args.draft else CFG['dpi'],'.png'),(170,'_preview.png')]:
        pix=document[0].get_pixmap(dpi=dpi,alpha=False)
        pix.save(HERE/(basename+suffix))
    if not args.draft:
        full=Image.open(HERE/(basename+'.png')).convert('RGB')
        full.save(HERE/(basename+'.tiff'),compression='tiff_lzw',dpi=(CFG['dpi'],CFG['dpi']))
    extracted=document[0].get_text();document.close()
    dr.to_csv(HERE/'source_DrugReflector_all100.csv',index=False)
    scores.to_csv(HERE/'source_primary_state_scores.csv',index=False)
    mat.to_csv(HERE/'source_heatmap_values.csv')
    provenance={
        'draft':args.draft,'figure_size_mm':[CFG['width_mm'],CFG['height_mm']],
        'input_score_file':str(score_path),'input_score_sha256':sha(score_path),
        'ranking_sha256':sha(HERE/CFG['drugreflector_source']),
        'molecular_assets_sha256':sha(HERE/CFG['asset_manifest']),
        'placements':placements,'pdf_text_characters':len(extracted),
        'no_md_results':True,'all100_candidates_plotted':True,
        'heatmap_statistic':'Median of mode1 Vina score across three technical seeds; R/S separate',
        'legend_numbers_rounding':'Two decimals only for display; source retains full precision',
        'image_processing':'Only blank margin crops and isotropic fit; actual ray rendered structural assets',
        'connector_revision':'Four right-pointing arrows centered between visible molecular edges at connector height; molecular image placements and arrow styles unchanged',
        'arrow_placements':arrows,
    }
    (HERE/(basename+'_render_checks.json')).write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    assert max(p['aspect_error']for p in placements)<1e-10
    assert len(extracted)>100,'Quantitative PDF text should remain selectable'
    print(pdf)


if __name__=='__main__': main()
