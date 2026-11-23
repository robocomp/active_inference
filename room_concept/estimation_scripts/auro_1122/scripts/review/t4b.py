import os,sys,math
WT='/home/pbustos/robocomp/components/webots-shadow/worlds/piso/threerooms-piso.wbt'
CM=os.path.abspath('world/threerooms-piso.wbt')
sys.path.insert(0,'/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/scripts')
which=sys.argv[1]
os.environ['WBT_PATH']=WT if which=='w2' else CM
if which=='w2': os.environ['C3A_POSE_OVERRIDE']='chair(3)=-7.4323,8.5130'
import c3a_objects as C
GT=C.gt_objects(os.environ['WBT_PATH'])
pts={'old':[('c3a_1','chair_94',-5.91,1.99,'chair'),('c3a_1','chair_96',-5.85,2.0,'chair'),('c3a_1','chair_98',0.85,9.52,'chair'),('c3a_1','nightstand_3',-4.04,13.24,'nightstand'),('c3a_1','table_14',-4.02,12.99,'table'),('c3a_2','table_17',-1.91,4.43,'table'),('c3a_3','chair_106',-5.91,2.01,'chair')],
'w2':[('c3a_w2_1','nightstand_5',-4.1,13.24,'nightstand'),('c3a_w2_2','chair_214',-1.29,2.4,'chair'),('c3a_w2_2','chair_216',-7.27,8.52,'chair'),('c3a_w2_2','nightstand_7',-4.11,13.24,'nightstand'),('c3a_w2_2','bottle_29(birth-pos world, c3a_w2_2)',-6.54,5.70,'bottle'),('c3a_w2_3','bottle_55(birth-pos world)',-6.55,5.71,'bottle')]}[which]
for run,n,x,y,c in pts:
    best=sorted((math.dist((x,y),(g[2],g[3])),g[0],g[1]) for g in GT)[:2]
    bc=min((math.dist((x,y),(g[2],g[3])),g[0]) for g in GT if g[1]==c)
    print(run,n,(x,y),'nearest same-class GT',round(bc[0],2),bc[1],'| nearest any',[(round(d,2),nm,cl) for d,nm,cl in best])
