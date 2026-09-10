
#create new tab for claude room
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/robocomp/components/active_inference/room_concept"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "claude room"

#create new tab for robot
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/robocomp/components/active_inference/robot_concept"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "robot"

#create new tab for room
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/robocomp/components/active_inference/room_concept"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "room"

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
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/robocomp/components/active_inference/viewer3d"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "controller"

#create new tab for viewer3d
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/robocomp/components/active_inference/viewer3d"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "viewer3d"

#create new tab for claude retina
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/robocomp/components/active_inference/retina"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "claude retina"

#create new tab for cortex
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/robocomp/components/cortex/build"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "cortex"

#create new tab for claude tesis
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/drive/Tesis/Noe"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "claude tesis"

#create new tab for claude-self
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/robocomp/components/active_inference/room_concept"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "claude-self"
