import os
from collections import namedtuple
from dataclasses import dataclass
from datetime import time
from typing import Any, Dict, List, Optional, Set
from types import MappingProxyType
from toposort import toposort

from fedml.workflow.jobs import Job, JobStatus
import time
from fedml.workflow.workflow_mlops_api import WorkflowMLOpsApi, WorkflowType, WorkflowStatus


@dataclass(frozen=True, eq=True, order=True)
class Node:
    name: str
    job: Job

    def __repr__(self):
        return f"Node(name={self.name}, job={self.job})"


@dataclass(frozen=True, eq=True, order=True)
class Metadata:
    nodes: Set[Node]
    topological_order: Any
    graph: Any

    def __repr__(self):
        return f"Metadata(nodes={self.nodes}, topological_order={self.topological_order}, graph={self.graph})"


@dataclass(frozen=True, eq=True, order=True)
class AdjacencyList:
    job: Job
    dependencies: List[Job]

    def __repr__(self):
        return f"AdjacencyList(job={self.job}, dependencies={self.dependencies})"


class Workflow:
    """
    Initialize the Workflow instance.

    Parameters:
    - loop (bool): Whether the workflow should loop continuously.
    """

    def __init__(self, name, loop: bool = False, api_key: str = None,
                 workflow_type: WorkflowType = WorkflowType.WORKFLOW_TYPE_DEPLOY):
        self.name: str = name
        self._metadata: Metadata | None = None
        self._loop: bool = loop
        self.jobs: Dict[str, AdjacencyList] = dict()
        self.input: Dict[Any, Any] = dict()
        self.api_key = api_key

        self.workflow_type = workflow_type
        self.id = None

    @property
    def metadata(self):
        return self._metadata if self._metadata else None

    @property
    def loop(self):
        return self._loop

    @metadata.setter
    def metadata(self, value):
        if not self._metadata:
            self._metadata = value
        else:
            raise ValueError("Workflow metadata cannot be modified.")

    @loop.setter
    def loop(self, value: bool):
        self._loop = value

    def add_job(self, job: Job, dependencies: Optional[List[Job]] = None):
        """
         Add a job to the workflow with optional dependencies.

         Parameters:
         - job (Job): An instance of the Job class.
         - dependencies (list): A list of Job instances that this job depends on.
         - Note that the order of the dependencies is important. The workflow will only run if it is able to resolve
         - dependencies and no cyclic dependencies exist. Workflows can be looped by setting the loop parameter to True.
         """

        if not isinstance(job, Job):
            raise TypeError("Only instances of the Job class (or its subclasses) can be added to the workflow.")

        if dependencies is None:
            dependencies = []

        if not all(isinstance(dep, Job) for dep in dependencies):
            raise TypeError("Dependencies must be instances of the Job class (or its subclasses).")

        if dependencies is None:
            dependencies = []

        if job.name in self.jobs:
            raise ValueError(f"Job {job.name} already exists in workflow.")

        job.workflow_id = self.id
        job.dependencies = dependencies
        self.jobs[job.name] = AdjacencyList(job=job, dependencies=dependencies)

    def run(self):
        """
        Run the workflow, executing jobs in the specified order.
        """

        WorkflowMLOpsApi.update_workflow(workflow_id=self.id, workflow_status=WorkflowStatus.RUNNING, api_key=self.api_key)

        self._compute_workflow_metadata()
        first_run = True
        has_set_first_input = False
        while first_run or self.loop:
            first_run = False
            for nodes in self.metadata.topological_order:
                jobs = [node.job for node in nodes]
                if not has_set_first_input and len(jobs) > 0:
                    jobs[0].set_inputs(self.input)
                    has_set_first_input = True
                self._execute_and_wait(jobs)

        WorkflowMLOpsApi.update_workflow(workflow_id=self.id, workflow_status=WorkflowStatus.FINISHED, api_key=self.api_key)

    def _execute_and_wait(self, jobs: List[Job]):
        """
        Execute the jobs and wait for them to complete.

        Parameters:
        - jobs (list): A list of Job instances to execute.
        """

        for job in jobs:
            dependencies = self.jobs.get(job.name).dependencies
            for dep in dependencies:
                job.append_input(dep.name, dep.get_outputs())

            job.run()

        while True:
            all_completed = True
            any_errored = False
            errored_jobs = []

            for job in jobs:
                status = job.status()
                if status != JobStatus.FINISHED:
                    all_completed = False

                    if status == JobStatus.FAILED or status == JobStatus.UNDETERMINED:
                        any_errored = True
                        errored_jobs.append(job.name)

            if all_completed:
                return True

            if any_errored:
                self._kill_jobs(jobs)
                WorkflowMLOpsApi.update_workflow(workflow_id=self.id, workflow_status=WorkflowStatus.FAILED, api_key=self.api_key)

                raise ValueError(f"Following jobs errored out, hence workflow cannot be completed: {errored_jobs}."
                                 "Please check the logs for more information.")

            time.sleep(0.1)

    def _kill_jobs(self, jobs: List[Job]):
        """
        Kill the jobs.

        Parameters:
        - jobs (list): A list of Job instances to kill.
        """
        for job in jobs:
            job.kill()

    def _compute_workflow_metadata(self):
        if self.metadata:
            raise ValueError("Workflow metadata already exists. This is not expected. Please report this issue.")

        node_dict = dict()
        graph = dict()

        for job_name, adjacency_list in self.jobs.items():
            node = node_dict.setdefault(job_name, Node(name=job_name, job=adjacency_list.job))
            graph.setdefault(node, set())

            for dependency in adjacency_list.dependencies:
                dependency_node = node_dict.setdefault(dependency.name, Node(name=dependency.name, job=dependency))
                graph[node].add(dependency_node)

        self.metadata = Metadata(nodes=set(node_dict.values()),
                                 graph=MappingProxyType(graph),
                                 topological_order=tuple(toposort(graph)))

        return self.metadata

    def get_job_status(self, job_name):
        for nodes in self.metadata.topological_order:
            jobs = [node.job for node in nodes]
            for job in jobs:
                if job_name == job.name:
                    return job.status()

        return JobStatus.UNDETERMINED

    def get_workflow_status(self):
        all_success = True
        has_failed = False
        all_completed = True
        for nodes in self.metadata.topological_order:
            jobs = [node.job for node in nodes]
            for job in jobs:
                status = job.status()
                if status == JobStatus.FINISHED:
                    pass
                elif status == JobStatus.FAILED:
                    has_failed = True
                    all_success = False
                else:
                    all_completed = False
                    all_success = False

        if all_completed and all_success:
            return JobStatus.FINISHED

        if all_completed and has_failed:
            return JobStatus.FAILED

        return JobStatus.RUNNING

    def set_workflow_input(self, input: Dict[Any, Any]):
        self.input = input

    def get_workflow_output(self):
        job_list = list()
        for nodes in self.metadata.topological_order:
            job_list.extend([node.job for node in nodes])

        return job_list[-1].get_outputs()

    def get_all_jobs_outputs(self):
        output_dicts = dict()
        job_list = list()
        for nodes in self.metadata.topological_order:
            job_list.extend([node.job for node in nodes])

        for job in job_list:
            output_dicts[job.name] = job.get_outputs()

        return output_dicts

    @staticmethod
    def get_workflow(workflow_name=None):
        workflow_name = os.environ.get("FEDML_CURRENT_WORKFLOW") if workflow_name is None else workflow_name
        return Workflow(workflow_name)

    def deploy(self):
        self.id = WorkflowMLOpsApi.create_workflow(
            workflow_name=self.name, workflow_type=self.workflow_type, api_key=self.api_key)
        if not self.id:
            raise Exception("Failed to deploy the workflow, unable to upload workflow metadata to the backend.")

        for job_name, adjacency_list in self.jobs.items():
            dependency_list = list()
            for dependency in adjacency_list.dependencies:
                dependency_list.append(dependency.name)

            adjacency_list.job.workflow_id = self.id
            adjacency_list.job.dependencies = dependency_list

            result = WorkflowMLOpsApi.add_run(
                workflow_id=self.id, job_name=job_name, run_id=None,
                dependencies=dependency_list, api_key=self.api_key
            )
            if not result:
                WorkflowMLOpsApi.update_workflow(workflow_id=self.id, workflow_status=WorkflowStatus.FAILED, api_key=self.api_key)
                raise Exception("Failed to deploy the workflow, unable to add job metadata to the backend.")

            try:
                adjacency_list.job.update_run_metadata()
            except Exception as e:
                WorkflowMLOpsApi.update_workflow(workflow_id=self.id, workflow_status=WorkflowStatus.FAILED, api_key=self.api_key)
                raise e



