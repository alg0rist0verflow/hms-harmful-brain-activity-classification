# hms-harmful-brain-activity-classification
Kaggle challenge hms-harmful-brain-activity-classification

# fix docker if required
```
sudo usermod -aG docker $USER
newgrp docker
docker run hello-world
```

```docker run --gpus all -v /media/miki/Storage/eeg/:/eeg/ --rm -it gcr.io/kaggle-gpu-images/python /bin/bash```