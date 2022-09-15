# Setup and configuration

Runners are controlled and only spawned by the [Actions-Runner-Controller](https://github.com/actions-runner-controller/actions-runner-controller) (ARC), so they will not show up in Github's Runner setting while idling.

The main backend software can only be installed by the clusters' Admins.
However, users in the `project-ns-opengptx` namespace can configure the controller using normal k8s deployment yaml in the below session.

Authentication for runners are done using Github-app as instructed in the ARC repo.


# Deployment files for running github actions on k8s cluster

`arc_runner_deployment.yaml` deploys runner managed by [Actions-Runner-Controller](https://github.com/actions-runner-controller/actions-runner-controller) (ARC). These runners are only created when need, thus does not permanently block resource on the cluster.

`unmanaged_runner_deployment.yaml` is the simplest way to deploy a runner. 
However, this is not recommended for runners with GPU access, because these runners will permanently block/occupy GPU on the cluster.

To deploy runner:
```bash
kubectl -f arc_runner_deployment.yaml
```
