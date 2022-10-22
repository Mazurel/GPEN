# GPEN for face relightening

This repo contains code for my Engineering thesis called.

It is based on: https://github.com/yangxy/GPEN

## How it works

TODO

## Usage

This repo contains wrapper file called `main.py`.
Everything is intended to be started from this file.

### Setting up the environemnt

To setup the environment required for training you should use provided docker file.
Docker image can also be pulled from: https://hub.docker.com/repository/docker/mazurel/gpen

Docker image is useful for example for deploying training onto: https://vast.ai/

**NOTE**: Remember to pull from the repos after loading docker image, as it may contain outdated code.

#### Downloading FFHQ dataset

When using docker image, `ffhq-dataset` repo should already be ready. It contains modifications, so that you can use [pydrive](https://pythonhosted.org/PyDrive/quickstart.html) so that you can download all of it in one go (without Google timeouting you). To use it, follow [this link](https://pythonhosted.org/PyDrive/quickstart.html).

Steps to get the dataset:

```bash
cd /workspace/ffhq-dataset
# Setup Google Drive client secrets here (or not use --pydrive)
python download_ffhq.py --json --images --pydrive --cmd_auth
# Move photos to /workspace/photos/. You cannot use one `mv` as there to many images, so it requires simple WA:
for dir in $(ls /workspace/ffhq-dataset/images1024x1024/); do mv /workspace/ffhq-dataset/images1024x1024/$dir/*.png /workspace/photos/; done
# Verify downloaded photos
python /workspace/GPEN/verify_photos.py /workspace/photos/
```
