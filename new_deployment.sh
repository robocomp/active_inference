
#create new tab for sub
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/robocomp/components/active_inference/utils"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "sub"

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

#create new tab for chat
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/robocomp/components/active_inference/chat_agent"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "chat"

#create new tab for viewer3d
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/robocomp/components/active_inference/viewer3d"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "viewer3d"

#create new tab for episodic
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/robocomp/components/active_inference"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "episodic"

#create new tab for cortex
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/robocomp/components/cortex/build"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "cortex"

#create new tab for claude-chat
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/robocomp/components/active_inference/chat_agent"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "claude-chat"

#create new tab for claude tesis
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/robocomp/components/active_inference"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "claude tesis"

#create new tab for claude waf paper
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/robocomp/components/active_inference"
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

#create new tab for montevideo
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/robocomp/components/active_inference.worktrees/course-montevideo"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "montevideo"

#create new tab for claude-retina
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/robocomp/components/active_inference/retina"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "claude-retina"

#create new tab for claude-em
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/robocomp/components/active_inference"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "claude-em"

#create new tab for claude_ltsm
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession
#get id of open session
sess0=`qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.activeSessionId`
#change to the saved working directory
qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.runCommand " cd /home/pbustos/robocomp/components/active_inference/ltsm_agent"
#change the title of session
qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle $sess0 "claude_ltsm"
