import models
import yaml
from yaml.loader import SafeLoader
import torch 
import collections 
import os
# Open the file and load the file
with open('final_models.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)

top_dir = "/home/tier3/jkrupa/public_html/zprlegacy/"
opset_version = 11
for key, val in data.items():
  
    if os.path.isfile("./final_models/"+key+".onnx"):
        print("Already have onnx model at /final_models/"+key+".onnx") 
        continue 
    print("working on model "+key)
    n_pf_dims = val["pf_features_dims"]
    n_sv_dims = val["sv_features_dims"]
    n_classes = val["num_classes"]
    n_particles = val["n_parts"]
    n_vertices = val["n_vertices"]
    event_branch = val["event_branch"] 

    if "ParticleNet" in val["model"]:
        conv_params = val["conv_params"]

        model = models.ParticleNetTagger(key, n_pf_dims, n_sv_dims, n_classes,  fc_params=val["fc_params"], event_branch=event_branch, conv_params=conv_params, for_inference=val["softmax"],sigmoid=val["sigmoid"] )
        model.load_state_dict(torch.load(top_dir+val["path"]))
        model.eval()

        
        pf_inputs = torch.randn(1,n_pf_dims,n_particles) 
        pf_mask = torch.randn(1,1,n_particles) 
        pf_points = torch.randn(1,2,n_particles) 
        sv_inputs = torch.randn(1,n_sv_dims,n_vertices)
        sv_mask = torch.randn(1,1,n_vertices) 
        sv_points = torch.randn(1,2,n_vertices) 
    
        inputs = (pf_points,pf_inputs,pf_mask,sv_points,sv_inputs,sv_mask)
        input_names = ["pf_points","pf_inputs","pf_mask","sv_points","sv_inputs","sv_mask"]
        if event_branch:
            input_names += ["event_features"]
            n_singletons = val["n_singletons"]
            e_inputs  = torch.randn(1,n_singletons)
            inputs += (e_inputs,) 

        torch.onnx.export(model,
                          inputs ,
                          "./final_models/"+key+".onnx", 
                          export_params=True,
                          opset_version=opset_version,         
                          do_constant_folding=True,
                          input_names=input_names,
                          output_names=["outputs"],
                          dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}},
    )

    if "IN" in val["model"]:
        n_singletons = val["n_singletons"]

        model = models.GraphNetv2(key, n_particles, n_classes, n_pf_dims, n_vertices=n_vertices, params_v=n_sv_dims, params_e=n_singletons, event_branch=event_branch,Do=val["Do"],De=val["De"],hidden=val["hidden"],pv_branch=n_sv_dims>0,softmax=val["softmax"],sigmoid=val["sigmoid"])
        model = model.to("cpu")
        device = torch.device('cpu')

        model.load_state_dict(torch.load(top_dir+val["path"],map_location=device))
        model.eval()
        pf_inputs = torch.randn(1,n_pf_dims,n_particles)
        sv_inputs = torch.randn(1,n_sv_dims,n_vertices)
        e_inputs  = torch.randn(1,n_singletons)

        inputs = (pf_inputs,sv_inputs)
        input_names = ["pf_inputs","sv_inputs"]
        if event_branch:
            inputs += (e_inputs,) 

            input_names += ["event_features"]

        torch.onnx.export(model,
                          inputs ,
                          "./final_models/"+key+".onnx", 
                          export_params=True,
                          opset_version=opset_version,         
                          do_constant_folding=True,
                          input_names=input_names,
                          output_names=["outputs"],
                          dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}},
    )

 
    del model
