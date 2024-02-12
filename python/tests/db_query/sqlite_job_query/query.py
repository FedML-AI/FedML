from fedml.computing.scheduler.model_scheduler import device_client_data_interface

# Check the client's job table
device_client_data_interface.FedMLClientDataInterface.get_instance().create_job_table()
job_list = device_client_data_interface.FedMLClientDataInterface.get_instance().get_jobs_from_db()

for job in job_list.job_list:
    print(type(job))
    job.show()
