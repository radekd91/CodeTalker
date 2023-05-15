

"""
Usage:
sh scripts/ensparc_bl.sh
specify condition and subject in the ensparc.yaml file
"""



#!/usr/bin/env python
import os
import cv2
import torch
import numpy as np
import librosa
import pickle
from pathlib import Path
from pydub import AudioSegment
from moviepy.editor import *

from transformers import Wav2Vec2Processor
from base.utilities import get_parser
from models import get_model
from base.baseTrainer import load_state_dict

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
cfg = get_parser()


import tempfile
from subprocess import call
os.environ['PYOPENGL_PLATFORM'] = 'osmesa' #egl
# if 'DISPLAY' not in os.environ or os.environ['DISPLAY'] == '':
#     os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrender
import trimesh
from psbody.mesh import Mesh


# The implementation of rendering is borrowed from VOCA: https://github.com/TimoBolkart/voca/blob/master/utils/rendering.py
def render_mesh_helper(args,mesh, t_center, rot=np.zeros(3), tex_img=None, z_offset=0):
    if args.dataset == "BIWI":
        camera_params = {'c': np.array([400, 400]),
                         'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                         'f': np.array([4754.97941935 / 8, 4754.97941935 / 8])}
    elif args.dataset == "vocaset":
        camera_params = {'c': np.array([400, 400]),
                         'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                         'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}

    frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}

    mesh_copy = Mesh(mesh.v, mesh.f)
    mesh_copy.v[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.v-t_center).T).T+t_center
    
    intensity = 2.0

    primitive_material = pyrender.material.MetallicRoughnessMaterial(
                alphaMode='BLEND',
                baseColorFactor=[0.3, 0.3, 0.3, 1.0],
                metallicFactor=0.8, 
                roughnessFactor=0.8 
            )


    tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f)
    tri_mesh.export("/is/cluster/scratch/kchhatre/Work/ENSPARC/baselines/CodeTalker/demo/output/test.obj")
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material,smooth=True)
  
    if args.background_black:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])
    else:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])
    
    camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                      fy=camera_params['f'][1],
                                      cx=camera_params['c'][0],
                                      cy=camera_params['c'][1],
                                      znear=frustum['near'],
                                      zfar=frustum['far'])

    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3,3] = np.array([0, 0, 1.0-z_offset])
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]])

    angle = np.pi / 6.0
    pos = camera_pose[:3,3]
    light_color = np.array([1., 1., 1.])
    light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

    light_pose = np.eye(4)
    light_pose[:3,3] = pos
    scene.add(light, pose=light_pose.copy())
    
    light_pose[:3,3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] =  cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], viewport_height=frustum['height'])
        color, _ = r.render(scene, flags=flags)
    except:
        print('pyrender: Failed rendering frame')
        color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return color[..., ::-1]

def main():
    global cfg
    model = get_model(cfg)
    # if torch.cuda.is_available():
    model = model.cuda()

    if os.path.isfile(cfg.model_path):
        print("=> loading checkpoint '{}'".format(cfg.model_path))
        checkpoint = torch.load(cfg.model_path, map_location=lambda storage, loc: storage.cpu())
        load_state_dict(model, checkpoint['state_dict'], strict=False)
        print("=> loaded checkpoint '{}'".format(cfg.model_path))
    else:
        raise RuntimeError("=> no checkpoint flound at '{}'".format(cfg.model_path))

    model.eval()
    # save_folder = cfg.demo_npy_save_folder
    # if not os.path.exists(save_folder):
    #     os.makedirs(save_folder, exist_ok=True)

    condition = cfg.condition
    subject = cfg.subject
    
    for i, p in enumerate(Path(cfg.demo_wav_path).glob('*')):
        sample_dir = p.stem
        out_path = os.path.join(cfg.demo_output_path, sample_dir)
        os.makedirs(out_path, exist_ok=True)
        print(f"Processing {sample_dir} ({i+1}/{len(list(Path(cfg.demo_wav_path).glob('*')))})")
        audio_fname = os.path.join(cfg.demo_wav_path, sample_dir)
    
        # for audio_file in Path(cfg.demo_wav_path).glob('*.m4a'):
        for audio_file in Path(audio_fname).glob('*.wav'):
            
            test_name = audio_file.stem
            # videoclip = VideoFileClip(str(audio_file)) # lrs3 are wav video files
            # audioclip = videoclip.audio
            # wav_dir = os.path.join(out_path, test_name, "audio")
            # os.makedirs(wav_dir, exist_ok=True)
            # wav_path = os.path.join(wav_dir, "audio.wav")
            # audioclip.write_audiofile(wav_path)
            
            wav_path = str(audio_file)
            
            # name = audio_file.stem
            # track = AudioSegment.from_file(audio_file, format="m4a")
            # _ = track.export(f"{Path(cfg.demo_wav_path) / name}.wav", format="wav")
            
            obj_dir = os.path.join(out_path, test_name, "meshes")
            os.makedirs(obj_dir, exist_ok=True)
            save_folder = os.path.join(out_path,test_name, "npy")
            os.makedirs(save_folder, exist_ok=True)
            # test4obj(model, f"{Path(cfg.demo_wav_path) / name}.wav", save_folder, condition, subject, obj_dir)
            test4obj(model, wav_path, save_folder, condition, subject, obj_dir)
        
    # for wav in Path(cfg.demo_wav_path).glob('*.wav'): wav.unlink()
    print("Done!")

def test(model, wav_file, save_folder, condition, subject):
    # generate the facial animation (.npy file) for the audio 
    print('Generating facial animation for {}...'.format(wav_file))
    
    template_file = os.path.join(cfg.data_root, cfg.template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')
    

    train_subjects_list = [i for i in cfg.train_subjects.split(" ")]
    one_hot_labels = np.eye(len(train_subjects_list))
    iter = train_subjects_list.index(condition)
    one_hot = one_hot_labels[iter]
    one_hot = np.reshape(one_hot,(-1,one_hot.shape[0]))
    one_hot = torch.FloatTensor(one_hot).to(device='cuda')

    temp = templates[subject]
    template = temp.reshape((-1))
    template = np.reshape(template,(-1,template.shape[0]))
    template = torch.FloatTensor(template).to(device='cuda')


    test_name = os.path.basename(wav_file).split(".")[0]
    if not os.path.exists(os.path.join(save_folder,test_name)):
        os.makedirs(os.path.join(save_folder,test_name))
    predicted_vertices_path = os.path.join(save_folder, test_name, 'condition_'+condition+'_subject_'+subject+'.npy')
    speech_array, _ = librosa.load(wav_file, sr=16000)
    processor = Wav2Vec2Processor.from_pretrained(cfg.wav2vec2model_path)
    audio_feature = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
    audio_feature = np.reshape(audio_feature,(-1,audio_feature.shape[0]))
    audio_feature = torch.FloatTensor(audio_feature).to(device='cuda')



    with torch.no_grad():
        prediction = model.predict(audio_feature, template, one_hot)
        prediction = prediction.squeeze() # (seq_len, V*3)
        np.save(predicted_vertices_path, prediction.detach().cpu().numpy())
        print(f'Save facial animation in {predicted_vertices_path}')


    ######################################################################################
    ##### render the npy file

    cfg.data_root = "/is/cluster/scratch/kchhatre/Work/ENSPARC/baselines/CodeTalker/vocaset"
    if cfg.dataset == "BIWI":
        template_file = os.path.join(cfg.data_root, "BIWI.ply")
    elif cfg.dataset == "vocaset":
        template_file = os.path.join(cfg.data_root, "FLAME_sample.ply")
         
    print("rendering: ", test_name)
    print("template_file: ", template_file)           
    template = Mesh(filename=template_file)
    predicted_vertices = np.load(predicted_vertices_path)
    predicted_vertices = np.reshape(predicted_vertices,(-1,cfg.vertice_dim//3,3))

    output_path = cfg.demo_output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    num_frames = predicted_vertices.shape[0]
    tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=output_path)
    
    writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), cfg.fps, (800, 800), True)
    center = np.mean(predicted_vertices[0], axis=0)

    for i_frame in range(num_frames):
        render_mesh = Mesh(predicted_vertices[i_frame], template.f)
        pred_img = render_mesh_helper(cfg,render_mesh, center)
        pred_img = pred_img.astype(np.uint8)
        writer.write(pred_img)

    writer.release()
    file_name = test_name+"_"+cfg.subject+"_condition_"+cfg.condition

    video_fname = os.path.join(output_path, file_name+'.mp4')
    cmd = ('ffmpeg' + ' -i {0} -pix_fmt yuv420p -qscale 0 {1}'.format(
       tmp_video_file.name, video_fname)).split()
    call(cmd)

    cmd = ('ffmpeg' + ' -i {0} -i {1} -vcodec h264 -ac 2 -channel_layout stereo -qscale 0 {2}'.format(
       wav_file, video_fname, video_fname.replace('.mp4', '_audio.mp4'))).split()
    call(cmd)

    if os.path.exists(video_fname):
        os.remove(video_fname)

def test4obj(model, wav_file, save_folder, condition, subject, obj_dir):
    wav_file = str(wav_file)
    # generate the facial animation (.npy file) for the audio 
    print('Generating facial animation for {}...'.format(wav_file))
    
    template_file = os.path.join(cfg.data_root, cfg.template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')
    

    train_subjects_list = [i for i in cfg.train_subjects.split(" ")]
    one_hot_labels = np.eye(len(train_subjects_list))
    iter = train_subjects_list.index(condition)
    one_hot = one_hot_labels[iter]
    one_hot = np.reshape(one_hot,(-1,one_hot.shape[0]))
    one_hot = torch.FloatTensor(one_hot).to(device='cuda')

    temp = templates[subject]
    template = temp.reshape((-1))
    template = np.reshape(template,(-1,template.shape[0]))
    template = torch.FloatTensor(template).to(device='cuda')

    # test_name = Path(obj_dir).stem
    # # test_name = os.path.basename(wav_file).split(".")[0]
    # if not os.path.exists(os.path.join(save_folder,test_name)):
    #     os.makedirs(os.path.join(save_folder,test_name))
    predicted_vertices_path = os.path.join(save_folder, 'condition_'+condition+'_subject_'+subject+'.npy')
    speech_array, _ = librosa.load(wav_file, sr=16000)
    processor = Wav2Vec2Processor.from_pretrained(cfg.wav2vec2model_path)
    audio_feature = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
    audio_feature = np.reshape(audio_feature,(-1,audio_feature.shape[0]))
    audio_feature = torch.FloatTensor(audio_feature).to(device='cuda')

    with torch.no_grad():
        prediction = model.predict(audio_feature, template, one_hot)
        prediction = prediction.squeeze() # (seq_len, V*3)
        np.save(predicted_vertices_path, prediction.detach().cpu().numpy())
        print(f'Save facial animation in {predicted_vertices_path}')


    ######################################################################################
    ##### render the npy file

    if cfg.dataset == "BIWI":
        template_file = os.path.join(cfg.data_root, "BIWI.ply")
    elif cfg.dataset == "vocaset":
        template_file = os.path.join(cfg.data_root, "FLAME_sample.ply")
         
    template = Mesh(filename=template_file)
    predicted_vertices = np.load(predicted_vertices_path)
    predicted_vertices = np.reshape(predicted_vertices,(-1,cfg.vertice_dim//3,3))

    # output_path = cfg.demo_output_path
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)

    num_frames = predicted_vertices.shape[0]
    center = np.mean(predicted_vertices[0], axis=0)

    for i_frame in range(num_frames):
        render_mesh = Mesh(predicted_vertices[i_frame], template.f)
        # obj_name = obj_dir / f"{i_frame}.obj"
        obj_name = os.path.join(obj_dir, f"{i_frame:05d}.obj")
        export_obj(obj_name,render_mesh, center)
    

def export_obj(obj_name, mesh, t_center, rot=np.zeros(3)):
    
    mesh_copy = Mesh(mesh.v, mesh.f)
    mesh_copy.v[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.v-t_center).T).T+t_center
    
    tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f)
    tri_mesh.export(obj_name)

if __name__ == '__main__':
    main()
    print("Changing permissions of the output files")
    os.system("find /is/cluster/fast/scratch/rdanecek/testing/enspark/baselines -type d -exec chmod 755 {} +")
