#!/usr/bin/env python3
"""Create synchronized source/result comparison previews from completed runs."""
from pathlib import Path
import json
import subprocess

ROOT=Path(__file__).resolve().parent
WEATHER_LABEL='雪天'
WEATHER_ENGLISH='SNOW'
OUT=ROOT/'evaluation/snow_v1'
SOURCE=Path('/root/sam31/data/dji_weather_rwy_standard/rwy_split2')
RUN=ROOT/'multiview/outputs/dji_snow_gpt6_v1/runs/gpt6_vllm_edge_v1'
VIEWS=[('LEFT','rwy_split2_left_left_original'),('FRONT','rwy_split2_v_front_left_original'),('REAR','rwy_split2_v_rear_left_original'),('UP','rwy_split2_v_up_left_original')]

def run(args):
    subprocess.run(args,check=True)

def encode_inputs(inputs):
    args=['ffmpeg','-hide_banner','-loglevel','error','-n','-filter_complex_threads','2']
    for path in inputs:
        if not path.is_file(): raise FileNotFoundError(path)
        args+=['-threads','2','-i',str(path)]
    return args

def tail(path,frames):
    return ['-map','[out]','-frames:v',str(frames),'-an','-c:v','libx264','-threads','4','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]

def filt(index,label,size):
    return f'[{index}:v]trim=end_frame=80,setpts=PTS-STARTPTS,scale={size}:{size},setsar=1,drawtext=text={label}:x=10:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.7[v{index}]'

def probe(path,frames,width,height):
    meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=codec_name,width,height,avg_frame_rate,nb_read_frames,duration','-of','json',str(path)],text=True))['streams'][0]
    assert int(meta['nb_read_frames'])==frames,meta
    assert meta['width']==width and meta['height']==height,meta
    assert meta['avg_frame_rate']=='16/1',meta
    assert meta['codec_name']=='h264',meta
    return meta

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=[];labels=[]
    for label,name in VIEWS:
        inputs += [SOURCE/(name+'.mp4'),RUN/('rwy_split2__'+name)/'vision.mp4']
        labels += [label+' - SOURCE',label+' - '+WEATHER_ENGLISH]
    overview=OUT/(WEATHER_LABEL+'_四视角原片对比.mp4')
    detail=OUT/(WEATHER_LABEL+'_逐视角原片对比.mp4')
    if not overview.exists():
        filters=[filt(i,label,512) for i,label in enumerate(labels)]
        layout='0_0|512_0|1024_0|1536_0|0_512|512_512|1024_512|1536_512'
        filters.append(''.join(f'[v{i}]' for i in range(8))+f'xstack=inputs=8:layout={layout}:shortest=1[out]')
        run(encode_inputs(inputs)+['-filter_complex',';'.join(filters)]+tail(overview,80))
    if not detail.exists():
        filters=[filt(i,label,640) for i,label in enumerate(labels)]
        for i in range(4):filters.append(f'[v{i*2}][v{i*2+1}]hstack=inputs=2:shortest=1[p{i}]')
        filters.append('[p0][p1][p2][p3]concat=n=4:v=1:a=0[out]')
        run(encode_inputs(inputs)+['-filter_complex',';'.join(filters)]+tail(detail,320))
    report={}
    for video,frames,width,height in [(overview,80,2048,1024),(detail,320,1280,640)]:
        report[video.name]=probe(video,frames,width,height)
        cover=video.with_suffix('.jpg')
        if not cover.exists():run(['ffmpeg','-hide_banner','-loglevel','error','-n','-ss','2','-i',str(video),'-frames:v','1',str(cover)])
    report['comparison']='Same 0..79 frame indices at native 16 fps; source vs completed GPT6-edited-prompt Cosmos output; detail order LEFT/FRONT/REAR/UP, five seconds each.'
    (OUT/'comparison_metadata.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':str(OUT),'validation':report},ensure_ascii=False))

if __name__=='__main__': main()
