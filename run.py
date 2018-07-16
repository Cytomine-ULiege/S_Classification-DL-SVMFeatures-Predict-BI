import os
from pathlib import Path

from cytomine import CytomineJob
import numpy as np
import pickle
from cytomine.models import Annotation, AlgoAnnotationTerm, AttachedFile, Job, AttachedFileCollection, \
    AnnotationCollection

from keras_util import MODEL_RESNET50, MODEL_VGG19, MODEL_VGG16, MODEL_INCEPTION_V3, MODEL_INCEPTION_RESNET_V2, \
    MODEL_MOBILE, MODEL_DENSE_NET_201, MODEL_NASNET_LARGE, MODEL_NASNET_MOBILE, PretrainedModelFeatures, ImageLoader, \
    batch_transform
from cytomine_util import parse_list_or_none, get_annotations


def main(argv):
    with CytomineJob.from_cli(argv) as cj:
        # prepare paths
        working_path = str(Path.home())
        data_path = os.path.join(working_path, "pred_data")
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        model_filename = "model.pkl"

        cj.job.update(progress=5, statusComment="Download model ...")
        model_job = Job().fetch(cj.parameters.cytomine_model_job_id)
        attached_files = AttachedFileCollection(model_job).fetch_with_filter("project", cj.project.id)
        if not (0 < len(attached_files) < 2):
            raise ValueError("More or less than 1 file attached to the Job (found {} file(s)).".format(len(attached_files)))
        attached_file = attached_files[0]
        if attached_file.filename != model_filename:
            raise ValueError("Expected model file name is '{}' (found: '{}').".format(model_filename, attached_file.filename))
        model_path = os.path.join(working_path, model_filename)
        attached_file.download(model_path)

        # load model
        with open(model_path, "rb") as file:
            data = pickle.load(file)
            model = data["model"]
            classifier = data["classifier"]
            network = data["network"]
            reduction = data["reduction"]

        # load and dump annotations
        cj.job.update(progress=10, statusComment="Download annotations.")
        annotations = get_annotations(
            project_id=cj.parameters.cytomine_project_id,
            images=parse_list_or_none(cj.parameters.cytomine_images_ids),
            users=parse_list_or_none(cj.parameters.cytomine_users_ids),
            showWKT=True
        )

        cj.job.update(statusComment="Fetch crops.", progress=15)
        n_samples = len(annotations)
        x = np.zeros([n_samples], dtype=np.object)
        for i, annotation in cj.monitor(enumerate(annotations), start=15, end=40, prefix="Fetch crops", period=0.1):
            file_format = os.path.join(data_path, "{id}.png")
            if not annotation.dump(dest_pattern=file_format):
                raise ValueError("Download error for annotation '{}'.".format(annotation.id))
            x[i] = file_format.format(id=annotation.id)

        available_nets = {
            MODEL_RESNET50, MODEL_VGG19, MODEL_VGG16, MODEL_INCEPTION_V3,
            MODEL_INCEPTION_RESNET_V2, MODEL_MOBILE, MODEL_DENSE_NET_201,
            MODEL_NASNET_LARGE, MODEL_NASNET_MOBILE
        }

        if network not in available_nets:
            raise ValueError("Invalid value (='{}'} for parameter 'network'.".format(network))
        if reduction not in {"average_pooling"}:
            raise ValueError("Invalid value (='{}') for parameter 'reduction'.".format(reduction))
        if classifier not in {"svm"}:
            raise ValueError("Invalid value (='{}') for parameter 'classifier'.".format(classifier))

        # prepare network
        cj.job.update(statusComment="Load neural network '{}'".format(network), progress=40)
        features = PretrainedModelFeatures(
            model=network, layer="last",
            reduction=reduction , weights="imagenet"
        )
        height, width, _ = features._get_input_shape(network)
        loader = ImageLoader(load_size_range=(height, height), crop_size=height, random_crop=False)

        cj.job.update(statusComment="Transform features.", progress=50)
        x_feat = batch_transform(loader, features, x, logger=cj.logger(start=50, end=70, period=0.1), batch_size=128)

        cj.job.update(statusComment="Prediction with '{}'.".format(classifier), progress=70)
        if hasattr(model, "n_jobs"):
            model.n_jobs = cj.parameters.n_jobs

        probas = None
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(x_feat)
            y_pred = model.classes_.take(np.argmax(probas, axis=1), axis=0)
        else:
            y_pred = model.predict(x_feat)

        cj.job.update(statusComment="Upload annotations.", progress=80)
        annotation_collection = AnnotationCollection()
        for i, annotation in cj.monitor(enumerate(annotations), start=80, end=100, period=0.1, prefix="Upload annotations"):
            annotation_collection.append(Annotation(
                location=annotation.location, id_image=annotation.image, id_project=annotation.project,
                term=[int(y_pred[i])], rate=float(probas[i]) if probas is not None else 1.0
            ).save())
        annotation_collection.save()

        cj.job.update(statusComment="Finished.", progress=100)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
