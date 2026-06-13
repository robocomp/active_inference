/*
 *    Copyright (C) 2026 by YOUR NAME HERE
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    RoboComp is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */


/** \mainpage RoboComp::room_concept
 *
 * \section intro_sec Introduction
 *
 * The room_concept component...
 *
 * \section interface_sec Interface
 *
 * interface...
 *
 * \section install_sec Installation
 *
 * \subsection install1_ssec Software depencences
 * ...
 *
 * \subsection install2_ssec Compile and install
 * cd room_concept
 * <br>
 * cmake . && make
 * <br>
 * To install:
 * <br>
 * sudo make install
 *
 * \section guide_sec User guide
 *
 * \subsection config_ssec Configuration file
 *
 * <p>
 * The configuration file etc/config...
 * </p>
 *
 * \subsection execution_ssec Execution
 *
 * Just: "${PATH_TO_BINARY}/room_concept --Ice.Config=${PATH_TO_CONFIG_FILE}"
 *
 * \subsection running_ssec Once running
 *
 * ...
 *
 */
#include <signal.h>

// QT includes
#include <QtCore>
#include <QtWidgets>

// ICE includes
#include <Ice/Ice.h>
#include <IceStorm/IceStorm.h>
#include <Ice/Application.h>

#include <ConfigLoader/ConfigLoader.h>

#include <sigwatch/sigwatch.h>
#include <sstream>

#include "genericworker.h"
#include "../src/specificworker.h"

#include <joystickadapterI.h>

#include <JoystickAdapter.h>


#define USE_QTGUI

#define PROGRAM_NAME    "room_concept"
#define SERVER_FULL_NAME   "RoboComp room_concept::room_concept"

namespace
{
QString ice_exception_to_qstring(const Ice::Exception& ex)
{
	std::ostringstream stream;
	stream << ex;
	return QString::fromStdString(stream.str());
}
}


template <typename SubInterfaceType>
void subscribe( const Ice::CommunicatorPtr& communicator,
                const IceStorm::TopicManagerPrxPtr& topicManager,
                const std::string& endpointConfig,
                std::string name_topic,
                const std::string& topicBaseName,
                SpecificWorker* worker,
                int index,
                std::shared_ptr<IceStorm::TopicPrx>& topic,
                Ice::ObjectPrxPtr& proxy, 
                const std::string& programName)
{
    try   
    {  
        if (!name_topic.empty()) name_topic += "/";
        name_topic += topicBaseName;

        Ice::ObjectAdapterPtr adapter = communicator->createObjectAdapterWithEndpoints(name_topic, endpointConfig);
        auto servant = std::make_shared<SubInterfaceType>(worker, index);
		proxy = adapter->addWithUUID(servant)->ice_oneway();

		qInfo() << QStringLiteral("%1 topic %2 will be used in subscription.")
											  .arg(QString::fromStdString(programName),
												   QString::fromStdString(name_topic));

        if(!topic)
        {
            try {
                topic = topicManager->create(name_topic);
				qWarning().noquote() << QStringLiteral("%1 topic %2 did not exist and was created.")
														 .arg(QString::fromStdString(programName),
															  QString::fromStdString(name_topic));
            }
            catch (const IceStorm::TopicExists&) {
                try{
					qWarning().noquote() << QStringLiteral("%1 topic %2 already exists; connecting to the existing topic.")
															 .arg(QString::fromStdString(programName),
																  QString::fromStdString(name_topic));
                    topic = topicManager->retrieve(name_topic);
                }
                catch(const IceStorm::NoSuchTopic&)
                {
					qCritical() << QStringLiteral("%1 topic %2 does not exist and could not be created.")
															  .arg(QString::fromStdString(programName),
																   QString::fromStdString(name_topic));
                    return;
                }
            }
            catch(const IceUtil::NullHandleException&)
            {
				qCritical() << QStringLiteral("%1 topic manager proxy is null.")
														  .arg(QString::fromStdString(programName));
                throw;
            }
            IceStorm::QoS qos;
            topic->subscribeAndGetPublisher(qos, proxy);
        }
        adapter->activate();
    }
    catch(const IceStorm::NoSuchTopic&)
    {
		qCritical() << "Error creating topic.";
    }
}


class room_concept : public Ice::Application
{
public:
	room_concept (QString configFile, QString prfx, bool startup_check) { 
		this->configFile = configFile.toStdString();
		this->prefix = prfx.toStdString();
		this->startup_check_flag=startup_check; 

		initialize();
		}

	Ice::InitializationData getInitializationDataIce();

private:
	void initialize();
	std::string prefix, configFile;
	ConfigLoader configLoader;
	TuplePrx tprx;
	bool startup_check_flag = false;

public:
	virtual int run(int, char*[]);
};

Ice::InitializationData room_concept::getInitializationDataIce(){
        Ice::InitializationData initData;
        initData.properties = Ice::createProperties();
        initData.properties->setProperty("Ice.Warn.Connections", this->configLoader.get<std::string>("Ice.Warn.Connections"));
        initData.properties->setProperty("Ice.Trace.Network", this->configLoader.get<std::string>("Ice.Trace.Network"));
        initData.properties->setProperty("Ice.Trace.Protocol", this->configLoader.get<std::string>("Ice.Trace.Protocol"));
        initData.properties->setProperty("Ice.MessageSizeMax", this->configLoader.get<std::string>("Ice.MessageSizeMax"));
		return initData;
}

void room_concept::initialize()
{
    this->configLoader.load(this->configFile);
	this->configLoader.printConfig();
	qInfo() << "Configuration loaded from" << QString::fromStdString(this->configFile);
}

int room_concept::run(int argc, char* argv[])
{
#ifdef USE_QTGUI
	QApplication a(argc, argv);  // GUI application
#else
	QCoreApplication a(argc, argv);  // NON-GUI application
#endif

	sigset_t sigs;
	sigemptyset(&sigs);
	sigaddset(&sigs, SIGHUP);
	sigaddset(&sigs, SIGINT);
	sigaddset(&sigs, SIGTERM);
	sigprocmask(SIG_UNBLOCK, &sigs, 0);

	UnixSignalWatcher sigwatch;
	sigwatch.watchForSignal(SIGINT);
	sigwatch.watchForSignal(SIGTERM);
	QObject::connect(&sigwatch, SIGNAL(unixSignal(int)), &a, SLOT(quit()));

	int status=EXIT_SUCCESS;

	std::shared_ptr<IceStorm::TopicPrx> joystickadapter_topic;
	Ice::ObjectPrxPtr joystickadapter;



	//Topic Manager code

	IceStorm::TopicManagerPrxPtr topicManager;
	try
	{
		topicManager = Ice::checkedCast<IceStorm::TopicManagerPrx>(communicator()->stringToProxy(configLoader.get<std::string>("Proxies.TopicManager")));
		if (!topicManager)
		{
		    qCritical() << "TopicManager.Proxy not defined in config file.";
		    qCritical() << "Config line example: TopicManager.Proxy=IceStorm/TopicManager:default -p 9999";
	        return EXIT_FAILURE;
		}
	}
	catch (const Ice::Exception &ex)
	{
		qCritical().noquote() << QStringLiteral("Exception: 'rcnode' not running: %1")
		                                      .arg(ice_exception_to_qstring(ex));
		return EXIT_FAILURE;
	}

	tprx = std::tuple<>();
	SpecificWorker *worker = new SpecificWorker(this->configLoader, tprx, startup_check_flag);
	QObject::connect(worker, SIGNAL(kill()), &a, SLOT(quit()));

	try
	{

		//Subscribe code
		subscribe<JoystickAdapterI>(communicator(),
		                    topicManager, configLoader.get<std::string>("Endpoints.JoystickAdapterTopic"),
						    configLoader.get<std::string>("Endpoints.JoystickAdapterPrefix"), "JoystickAdapter", worker,  0,
						    joystickadapter_topic, joystickadapter, PROGRAM_NAME);

		// Server adapter creation and publication
			qInfo() << SERVER_FULL_NAME << "started";

		// User defined QtGui elements ( main window, dialogs, etc )

		#ifdef USE_QTGUI
			//ignoreInterrupt(); // Uncomment if you want the component to ignore console SIGINT signal (ctrl+c).
			a.setQuitOnLastWindowClosed( true );
		#endif
		// Run QT Application Event Loop
		a.exec();

		// Avoid late IceStorm unsubscribe during process teardown; we already
		// perform component shutdown on aboutToQuit.


		status = EXIT_SUCCESS;
	}
	catch(const Ice::Exception& ex)
	{
		status = EXIT_FAILURE;

		qCritical() << QStringLiteral("Exception raised on main thread: %1")
		                                      .arg(ice_exception_to_qstring(ex));

	}
	#ifdef USE_QTGUI
		a.quit();
	#endif

	status = EXIT_SUCCESS;
	// Worker shutdown is executed from aboutToQuit via SpecificWorker::request_shutdown().
	// Avoid explicit deletion here to prevent late teardown crashes.
	return status;
}

int main(int argc, char* argv[])
{
	std::string arg;

	// Set config file
	QString configFile("etc/config");
	bool startup_check_flag = false;
	QString prefix("");
	if (argc > 1)
	{

		// Search in argument list for arguments
		QString startup = QString("--startup-check");
		QString initIC = QString("--Ice.Config=");
		QString prfx = QString("--prefix=");
		for (int i = 0; i < argc; ++i)
		{
			arg = argv[i];
			if (arg.find(startup.toStdString(), 0) != std::string::npos)
			{
				startup_check_flag = true;
				qInfo() << "Startup check enabled";
			}
			else if (arg.find(prfx.toStdString(), 0) != std::string::npos)
			{
				prefix = QString::fromStdString(arg).remove(0, prfx.size());
				if (prefix.size()>0)
					prefix += QString(".");
				qInfo().noquote() << QStringLiteral("Configuration prefix: <%1>").arg(prefix);
			}
			else if (arg.find(initIC.toStdString(), 0) != std::string::npos)
			{
				configFile = QString::fromStdString(arg).remove(0, initIC.size());
				qInfo() << "Starting with config file:" << configFile;
			}
			else if (i==1 and argc==2 and arg.find("--", 0) == std::string::npos)
			{
				configFile = QString::fromStdString(arg);
				qInfo() << "Starting with config file:" << configFile;
			}
		}

	}
	room_concept app(configFile, prefix, startup_check_flag);

	return app.main(argc, argv, app.getInitializationDataIce());
}
