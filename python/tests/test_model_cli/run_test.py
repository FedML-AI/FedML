import argparse
import json
import logging
import os
import sys
import time
import uuid
from os.path import expanduser

import fedml
from fedml.api.modules import model
from fedml.computing.scheduler.comm_utils import sys_utils
from fedml.computing.scheduler.comm_utils.job_monitor import JobMonitor
from fedml.computing.scheduler.model_scheduler.device_http_inference_protocol import FedMLHttpInference
from fedml.computing.scheduler.model_scheduler.device_http_proxy_inference_protocol import FedMLHttpProxyInference
from fedml.computing.scheduler.slave.client_constants import ClientConstants
from fedml.core.mlops.mlops_runtime_log import MLOpsRuntimeLog
from fedml.core.mlops.mlops_runtime_log_daemon import MLOpsRuntimeLogDaemon
from fedml.computing.scheduler.model_scheduler import device_client_constants
from fedml.computing.scheduler.scheduler_core.endpoint_sync_protocol import FedMLEndpointSyncProtocol
from fedml.computing.scheduler.comm_utils.container_utils import ContainerUtils


mqtt_config = dict()
mqtt_config["BROKER_HOST"] = "mqtt-dev.fedml.ai"
mqtt_config["BROKER_PORT"] = 1883
mqtt_config["MQTT_USER"] = "admin"
mqtt_config["MQTT_PWD"] = ""
mqtt_config["MQTT_KEEPALIVE"] = 180


def test_model_create_push(config_version="release"):
    cur_dir = os.path.dirname(__file__)
    model_config = os.path.join(cur_dir, "llm_deploy", "serving.yaml")
    model_name = f"test_model_{str(uuid.uuid4())}"
    fedml.set_env_version(config_version)
    model.create(model_name, model_config=model_config)
    model.push(
        model_name, api_key="",
        tag_list=[{"tagId": 147, "parentId": 3, "tagName": "LLM"}])


def test_cleanup_model_monitor_process():
    sys_utils.cleanup_model_monitor_processes(
        1627, "ep-1124-304-13ad33", "", "", "")


def test_log_endpoint_status():
    fedml.set_env_version("dev")

    endpoint_id = 1682
    fedml.set_env_version("dev")
    fedml.mlops.log_endpoint_status(
        endpoint_id, device_client_constants.ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED)


def test_show_hide_mlops_console_logs(args):
    args.log_file_dir = ClientConstants.get_log_file_dir()
    args.run_id = 0
    args.role = "client"
    client_ids = list()
    client_ids.append(111)
    args.client_id_list = json.dumps(client_ids)
    setattr(args, "using_mlops", True)
    MLOpsRuntimeLog.get_instance(args).init_logs(show_stdout_log=False)
    print("log 1")
    MLOpsRuntimeLog.get_instance(args).enable_show_log_to_stdout()
    print("log 2")
    MLOpsRuntimeLog.get_instance(args).enable_show_log_to_stdout(enable=False)
    print("log 3")


def test_log_lines_to_mlops():
    fedml.mlops.log_run_log_lines(
        1685, 0, ["failed to upload logs4"], log_source="MODEL_END_POINT")


def test_unify_inference():
    # Test inference
    from fedml.computing.scheduler.comm_utils.job_utils import JobRunnerUtils
    JobMonitor.get_instance().mqtt_config = mqtt_config
    model_url = "http://127.0.0.1:64755/predict"
    input_list = {"arr":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.0100005,-0.0100005,-0.0100005,-0.013973799,-0.0189315247,-0.023184301,-0.0360728861,-0.0392619154,-0.0380269994,-0.0390143887,-0.0346046778,-0.0257765396,-0.0209733754,-0.0217809993,-0.0144984527,-0.0118807892,0,0,0,0,0,0,0,0,0,0,0,0,-0.0178081425,-0.0232058779,-0.0298662898,-0.0414395151,-0.0586512813,-0.0812643979,-0.105997038,-0.121704878,-0.134457288,-0.139756261,-0.141562422,-0.135229133,-0.120246727,-0.104490087,-0.0870044931,-0.0716699334,-0.0485892545,-0.0324260775,-0.0216926329,-0.0100005,0,0,0,0,0,0,0,-0.0132956624,-0.0225936238,-0.0383702224,-0.0598206019,-0.0842014426,-0.118390816,-0.154266827,-0.188282524,-0.219803054,-0.242936317,-0.255020324,-0.259481423,-0.249404582,-0.226727106,-0.200418885,-0.16716117,-0.134317009,-0.0958717755,-0.0736565245,-0.0503983075,-0.0269783475,-0.0168919,-0.0100005,0,0,0,0,-0.0147795885,-0.025122101,-0.0381226487,-0.0786317321,-0.119593671,-0.165704529,-0.228814281,-0.288620224,-0.354491034,-0.421140618,-0.480243669,-0.527064646,-0.540807419,-0.521388017,-0.474446021,-0.403948632,-0.336571539,-0.271580657,-0.20666741,-0.154539645,-0.108856709,-0.0677589146,-0.0340327281,-0.0215091205,0,0,-0.0100005,-0.0107381289,-0.0260253876,-0.0570600482,-0.0914378767,-0.143000013,-0.199005834,-0.266034404,-0.353401549,-0.450251488,-0.551598332,-0.647939202,-0.743171364,-0.818162561,-0.851073275,-0.83112168,-0.763764496,-0.659992784,-0.547527626,-0.439376979,-0.33557659,-0.254856553,-0.183933732,-0.126755715,-0.0706477667,-0.0388818206,0,0,0,-0.0134176155,-0.0390612132,-0.0873974922,-0.133107017,-0.194532142,-0.27478633,-0.369886454,-0.482920333,-0.605294063,-0.735621386,-0.869509827,-0.989564738,-1.09132506,-1.13182948,-1.09408349,-0.996373436,-0.868781173,-0.717778845,-0.570649327,-0.439021868,-0.326889344,-0.235934504,-0.167697996,-0.0995100269,-0.0479392976,-0.0187851186,0,-0.0117322667,-0.0288274493,-0.0646532861,-0.118956716,-0.17783758,1.53795878,2.57176245,1.53212043,1.00392168,-0.179355647,-0.591732991,-1.05273662,-1.15378689,-1.22142979,-1.2388156,-1.21321586,-1.14302847,-1.02018313,-0.857098743,-0.676706697,-0.516203262,-0.379287244,-0.271402545,-0.189934521,-0.119940614,-0.0556340911,-0.0145752163,0,-0.0206611389,-0.0437166621,-0.0808756237,-0.140488164,-0.207699245,3.7747726,3.14033146,2.28939169,1.76127332,1.4318542,1.1313135,0.679164893,0.665484747,0.666043389,0.680680095,0.677305174,0.665508286,0.721340316,0.883661589,0.91751869,0.0282541074,-0.401002939,-0.283099723,-0.194831338,-0.123075256,-0.066612686,-0.0161462821,-0.0112546885,-0.0293918605,-0.0484646663,-0.093178326,-0.146682925,-0.218121209,0.830460131,1.04725853,0.147086928,0.259684517,0.495679969,0.998953721,1.29535061,1.12204782,1.41528197,1.4259952,1.36416372,1.22805443,1.03395727,1.40874227,1.73166837,1.00260058,-0.401823716,-0.275049233,-0.181713744,-0.107567122,-0.0566041118,-0.0189159236,-0.0121427928,-0.0243168731,-0.050270377,-0.0887358114,-0.138806025,-0.212706019,-0.321729999,-0.462313723,-0.652442841,-0.845524923,-0.961258323,-0.793125052,-0.226359955,-0.640468216,-0.12372009,-0.167157468,-0.255843161,-0.441448335,-0.792766628,1.30597044,1.81460411,0.691054579,-0.383665051,-0.26310513,-0.166473946,-0.0799663431,-0.0455007946,-0.0195541446,-0.0100005,-0.0186206584,-0.0414986832,-0.0722615997,-0.123238725,-0.212256343,-0.331309824,-0.491126078,-0.687704902,-0.86260267,-0.939124713,-0.869991467,-0.758168797,-0.722198511,-0.739826964,-0.809980626,-0.911188613,-1.00032001,-0.221550751,1.53134484,1.47605194,-0.273150738,-0.363157263,-0.252975575,-0.157152039,-0.0652009258,-0.0335283586,-0.0124209728,0,-0.014849279,-0.0329699917,-0.0601451792,-0.118353377,-0.219271688,-0.354392407,-0.523006773,-0.71568287,-0.862626101,-0.90524289,-0.831592288,-0.751312636,-0.762948163,-0.825877849,-0.930232292,-1.04727288,-0.879016953,1.11455708,1.61660969,0.264000765,-0.464282235,-0.354907482,-0.256014147,-0.158427696,-0.0620647188,-0.0242921899,0,0,-0.0117874599,-0.0252632841,-0.0502423656,-0.115068847,-0.235195531,-0.377531303,-0.547311188,-0.723069536,-0.848981953,-0.878897369,-0.826469482,-0.795496372,-0.883536617,-0.994814123,-1.13364619,-1.20871511,0.0000560198157,1.28700658,1.50082995,-0.122561277,-0.462110102,-0.360151562,-0.263898374,-0.166295096,-0.0568635009,-0.0105441394,0,0,0,-0.016636779,-0.0423254862,-0.119931644,-0.252550583,-0.39191634,-0.556171069,-0.717849905,-0.829516019,-0.854549188,-0.84598967,-0.889246054,-1.03761315,-1.16457617,-1.30025654,-0.740699086,1.05188993,1.3036988,-0.163440609,-0.59058464,-0.474233049,-0.368789557,-0.274082099,-0.174264813,-0.0696188843,-0.018003151,0,0,0,-0.0168610568,-0.0451688568,-0.131668459,-0.267838929,-0.398906806,-0.548202377,-0.690077015,-0.789823563,-0.831599129,-0.861314493,-0.95681566,-1.11036634,-1.22743073,-1.31006468,-0.02573686,1.14239899,0.761423491,-0.706825874,-0.608999426,-0.492457882,-0.380502867,-0.279282191,-0.173984018,-0.0767235054,-0.0195871373,-0.0100005,0,-0.0100005,-0.024817808,-0.0552275065,-0.148243512,-0.283202341,-0.4022125,-0.534598048,-0.656007943,-0.738083794,-0.781657503,-0.824620535,-0.918824463,-1.04078449,-1.13391454,-1.09212795,0.70592031,1.17679031,-0.37378182,-0.758547572,-0.62868064,-0.501492113,-0.381043892,-0.270505206,-0.168251255,-0.0784168728,-0.022799968,-0.0157856413,0,0,-0.0269850288,-0.0676999793,-0.167498207,-0.298089736,-0.411096027,-0.522810883,-0.625838621,-0.693423683,-0.731704263,-0.767086709,-0.82998003,-0.921590434,-1.00562716,0.0779492952,1.22959017,0.636500653,-0.901400043,-0.769630793,-0.635363773,-0.494618472,-0.369117095,-0.255794246,-0.156732083,-0.0783809414,-0.0267109338,-0.0148726634,0,-0.0100005,-0.0348385687,-0.0869311199,-0.185622432,-0.311777198,-0.427690033,-0.530457702,-0.612837575,-0.669073252,-0.706628103,-0.737178903,-0.779583917,-0.866698428,-0.288157768,1.2193059,1.10500698,-0.50413989,-0.909137779,-0.774520432,-0.619405771,-0.472096102,-0.344822207,-0.235626373,-0.144455008,-0.0769092863,-0.0286146987,-0.0100005,0,-0.0100005,-0.0342628198,-0.101174053,-0.195711272,-0.324606261,-0.442716711,-0.545960978,-0.637281741,-0.703742928,-0.753441795,-0.788772419,-0.829773267,-0.745526297,0.949893727,1.18293215,0.385795002,-1.023299,-0.89872884,-0.736858006,-0.575258663,-0.430322485,-0.30912025,-0.209889823,-0.13189517,-0.0731506415,-0.0276674735,-0.0100005,0,-0.0100005,-0.0400234981,-0.10709374,-0.194645695,-0.316981297,-0.440895564,-0.560086039,-0.667605659,-0.763806998,-0.843535003,-0.903604039,-0.938010529,0.763887624,1.12176928,0.784111,-0.818046093,-0.991046672,-0.828340182,-0.652780006,-0.495325185,-0.364891317,-0.261772085,-0.17529887,-0.112966586,-0.0617374486,-0.0270715466,0,0,0,-0.0406825662,-0.0978606438,-0.177848987,-0.287783481,-0.412614752,-0.543271605,-0.671018812,-0.798159188,-0.916686263,-1.02499517,-0.773682132,1.09355574,1.05041156,-0.498209852,-1.05256459,-0.870980804,-0.688431167,-0.523166414,-0.391308572,-0.282035183,-0.199071147,-0.13652517,-0.0893688913,-0.041317086,-0.016850831,0,0,0,-0.0283386899,-0.0765120563,-0.141969555,-0.232658498,-0.341261378,-0.469723228,-0.606194512,-0.747366354,-0.880786554,-0.729389144,0.895224865,1.11943124,-0.105438374,-1.00783177,-0.859696548,-0.683890026,-0.531181637,-0.395889778,-0.289956123,-0.203267966,-0.14295145,-0.0963532989,-0.0643914026,-0.0337070214,-0.0111853003,0,0,-0.0100005,-0.0151722732,-0.0480051146,-0.0951161616,-0.160643556,-0.245453283,-0.353245922,-0.474265429,-0.598667391,-0.729305101,0.389322873,1.38694264,1.37486731,-0.403963644,-0.77444593,-0.638730244,-0.502999283,-0.387339921,-0.279971294,-0.198381814,-0.135822721,-0.0965383286,-0.0633365644,-0.0427549534,-0.0257581657,-0.0100005,0,0,0,0,-0.0237543896,-0.0522032466,-0.0858749627,-0.140703979,-0.208515621,-0.290149335,-0.368567087,0.334201602,2.33307288,2.27286258,2.23777229,0.0412218057,-0.494890333,-0.422342015,-0.339048837,-0.257069088,-0.185534152,-0.136577185,-0.0860242391,-0.0578259874,-0.033636416,-0.0181122384,-0.0100005,0,0,0,0,0,-0.0136274661,-0.0285803164,-0.0474793553,-0.0779785591,-0.118532172,-0.167201555,-0.214787719,2.22171299,4.30500754,4.03125111,3.36505818,0.379953648,-0.284269948,-0.247694588,-0.205869945,-0.155925102,-0.116435448,-0.0857647974,-0.0546508166,-0.0401800073,-0.023758997,-0.0165780693,-0.0100005,0,0,0,0,0,0,-0.0115748833,-0.0284271584,-0.0506655656,-0.0740332846,-0.100455604,-0.124744578,4.17363552,7.81243004,5.7896979,0.322149281,-0.181506609,-0.160333393,-0.139182079,-0.118875455,-0.0873316648,-0.0700227708,-0.0540690537,-0.0384297037,-0.0265616274,-0.0161844507,-0.0119683967,0,0,0,0,0,0,0,0,0,-0.0132918601,-0.0159980455,-0.0207236291,-0.0266997366,-0.0284703819,-0.0343035092,-0.0410336906,-0.0488886427,-0.0548357917,-0.0551988782,-0.0469971082,-0.0388769026,-0.0316010302,-0.0285226846,-0.021736589,-0.0100005,0,0,0,0,0,0]}
    JobMonitor.get_instance().is_inference_ready(model_url, device_id=178076, endpoint_id=1870)
    JobMonitor.get_instance().inference(
        178076, 1870, model_url, input_list, [], timeout=2)


def test_http_inference():
    # Test http and http proxy inference
    endpoint_id = 383
    inference_url = "http://127.0.0.1:10000/predict"
    input = {"messages": [{"role": "user", "content": "What is a good cure for hiccups?"}]}
    output = []
    print(FedMLHttpProxyInference.run_http_proxy_inference_with_request(
        endpoint_id, inference_url, input, output))

    print(FedMLHttpInference.run_http_inference_with_curl_request(inference_url, input, output))


def test_http_inference_with_stream_mode():
    # Test http and http proxy inference
    endpoint_id = 383
    inference_url = "http://127.0.0.1:10000/predict"
    input = {
        "stream": True, "messages": [{"role": "system", "content": "You are a helpful assistant."},
                                     {"role": "user", "content": "Who won the world series in 2020?"},
                                     {"role": "assistant",
                                      "content": "The Los Angeles Dodgers won the World Series in 2020."},
                                     {"role": "user", "content": "Where was it played?"}],
        "model": "mythomax-l2-13b-openai-endpoint01"}
    output = []
    print(FedMLHttpProxyInference.run_http_proxy_inference_with_request(
        endpoint_id, inference_url, input, output))

    print(FedMLHttpInference.run_http_inference_with_curl_request(inference_url, input, output))


def test_create_container():
    import docker
    client = docker.from_env()
    inference_image_name = "fedml/fedml-default-inference-backend"
    default_server_container_name = "fedml-test-docker"
    port_inside_container = 2345
    usr_indicated_worker_port = None
    new_container = client.api.create_container(
        image=inference_image_name,
        name=default_server_container_name,
        # volumes=volumns,
        ports=[port_inside_container],  # port open inside the container
        entrypoint=["sleep", '1000'],
        # environment=environment,
        host_config=client.api.create_host_config(
            # binds=binds,
            port_bindings={
                port_inside_container: usr_indicated_worker_port  # Could be either None or a port number
            },
            # device_requests=device_requests,
            # mem_limit = "8g",   # Could also be configured in the docker desktop setting
        ),
        detach=True,
        # command=entry_cmd if enable_custom_image else None
    )
    client.api.start(container=new_container.get("Id"))

    # Get the port allocation
    cnt = 0
    while True:
        cnt += 1
        try:
            if usr_indicated_worker_port is not None:
                inference_http_port = usr_indicated_worker_port
                break
            else:
                # Find the random port
                port_info = client.api.port(new_container.get("Id"), port_inside_container)
                inference_http_port = port_info[0]["HostPort"]
                logging.info("inference_http_port: {}".format(inference_http_port))
                break
        except:
            if cnt >= 5:
                raise Exception("Failed to get the port allocation")
            time.sleep(3)


def test_get_endpoint_logs():
    JobMonitor.get_instance().monitor_endpoint_logs()


def test_check_endpoint_heath():
    container_name = "fedml_default_server_container__61cbbd11750b6683ce70295fbe3c5fe4"
    # stopped = ContainerUtils.get_instance().stop_container(container_name)
    started, inference_port = ContainerUtils.get_instance().start_container(container_name)
    #started, inference_port = ContainerUtils.get_instance().restart_container(container_name)
    master_id = 178077
    worker_device_id = 178076
    end_point_id = 1918
    end_point_name = "test-1221-3"
    model_name = "mnist"
    model_id = 375
    model_version = "v0-Thu Nov 16 09:41:42 GMT 2023"
    agent_config = dict()
    agent_config["mqtt_config"] = mqtt_config
    sync_proctol = FedMLEndpointSyncProtocol(agent_config=agent_config)
    sync_proctol.setup_client_mqtt_mgr()
    sync_proctol.setup_listener_for_sync_device_info(master_id)
    sync_proctol.send_sync_inference_info(
        master_id, worker_device_id, end_point_id, end_point_name, model_name,
        model_id, model_version, inference_port)
    while True:
        time.sleep(1)


def test_log_endpoint():
    os.environ["FEDML_CURRENT_RUN_ID"] = "2921"
    os.environ["FEDML_CURRENT_EDGE_ID"] = "74767"
    fedml.log_endpoint({"test_metric": 0.1})
    fedml.log({"test_metric": 0.2}, is_endpoint_metric=True)
    print("OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--version", "-v", type=str, default="dev")
    parser.add_argument("--log_file_dir", "-l", type=str, default="~")
    args = parser.parse_args()

    print("Hi everyone, I am testing the model cli.\n")

    test_log_endpoint()

    logging.getLogger().setLevel(logging.INFO)
    fedml.set_env_version("dev")

    test_check_endpoint_heath()

    time.sleep(1000000)
