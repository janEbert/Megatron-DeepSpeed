# Deployment files for running github actions on k8s cluster

`arc_runner_deployment.yaml` deploys runner managed by [Actions-Runner-Controller](https://github.com/actions-runner-controller/actions-runner-controller) (ARC). These runners are only created when need, thus does not permanently block resource on the cluster.

`unmanaged_runner_deployment.yaml` is the simplest way to deploy a runner. 
However, this is not recommended for runners with GPU access, because these runners will permanently block/occupy GPU on the cluster.

To deploy runner:
```bash
kubectl -f arc_runner_deployment.yaml
```
