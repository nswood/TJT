import sys
import utils
import argparse
import numpy as np
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='Variable importance.')
parser.add_argument('--file', action='store', type=str, help='path to file')
import mplhep as hep

args = parser.parse_args()

# Helper method to print importances and visualize distribution
def visualize_importances(feature_names, importances, title="Average Feature Importances", plot=True, axis_title="Features"):

    x_pos = (np.arange(len(feature_names)))
    plt.clf()
    fig,ax=plt.subplots()#plt.figure(figsize=(12,6))
    if importances.ndim>1:
        order=np.argsort(np.median(importances,axis=1))[::-1]
        ax.violinplot([x for x in importances[order]],positions=x_pos,showmedians=True,)
    else:
        order=np.argsort(importances)[::-1]
        ax.scatter(x_pos, importances[order])
    ax.set_xticks(x_pos,np.array(feature_names)[order],rotation=45)
    #ax.set_xticks(x_pos[order], np.array(feature_names)[order], rotation=45)
    ax.set_xlabel(axis_title, horizontalalignment='right',x=1.0,**utils.axis_font)
    ax.set_ylabel("Relative importance", horizontalalignment='right',y=1.0,**utils.axis_font)
    opath=args.file.split("/")[:-1]
    opath="/".join(opath)
    hep.cms.label("Preliminary",rlabel=utils.rlabel,data=False)
    process=args.file.split("/")[-1].replace("_score.npz","")
    ax.text(0.8,0.85,process+" score",transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(opath+process+axis_title.replace(" ","")+".png")
    plt.savefig(opath+process+axis_title.replace(" ","")+".pdf")
    plt.clf()
    


attr = np.load(args.file)
visualize_importances(utils._p_features_labels, attr["pf"][0][:,0],args.file,axis_title="PF features particle 1") 
visualize_importances(utils._p_features_labels, attr["pf"][0][:,-1],args.file,axis_title="PF features particle 60") 
visualize_importances(utils._p_features_labels, attr["pf"][0][:],args.file,axis_title="PF features all particles") 
visualize_importances(utils._SV_features_labels, attr["sv"][0][:,0],args.file,axis_title="SV features vertex 1") 
visualize_importances(utils._SV_features_labels, attr["sv"][0][:,-1],args.file,axis_title="SV features vertex 5") 
visualize_importances(utils._SV_features_labels, attr["sv"][0][:,],args.file,axis_title="SV features all vertices") 
visualize_importances(utils._singleton_features_labels, attr["event"][0],args.file,axis_title="Event features") 
