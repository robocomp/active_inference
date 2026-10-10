
#create new tab for room
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/robocomp/components/active_inference/room_concept"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "room"

#create new tab for residual
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/robocomp/components/active_inference/residual_concept"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "residual"

#create new tab for controller
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/robocomp/components/active_inference/controller"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "controller"

#create new tab for retina
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/robocomp/components/active_inference/retina"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "retina"

#create new tab for door
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/robocomp/components/active_inference/door_concept"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "door"

#create new tab for viewer3d
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/robocomp/components/active_inference/viewer3d"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "viewer3d"

#create new tab for ltsm
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/robocomp/components/active_inference/ltsm_agent"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "ltsm"

#create new tab for cortex
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/robocomp/components/cortex/build"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "cortex"

#create new tab for claude-self
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/robocomp/components/active_inference/room_concept"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "claude-self"

#create new tab for claude tesis
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/drive/Tesis/Noe"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "claude tesis"

#create new tab for claude waf paper
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/drive/Papers/Papers"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "claude waf paper"

#create new tab for hover
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/hoverboard"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "hover"

#create new tab for claude-lag
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/robocomp/components/active_inference/robot_concept"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "claude-lag"

#create new tab for montevideo
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/robocomp/components/active_inference/montevideo"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "montevideo"
