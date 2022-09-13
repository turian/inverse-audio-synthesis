#!/usr/bin/env python3
# https://community.wandb.ai/t/using-the-python-api-to-delete-models-with-no-tag-minimal/1498?u=turian
# "Rather than using api.artifact_versions, it uses the versions
# method on artifact_collection."

import wandb
from tqdm.auto import tqdm

# dry_run = True
dry_run = False
# api = wandb.Api(overrides={"project": "vicreg-synth1b1-pqmfs", "entity": "turian"})
api = wandb.Api()
project = api.project("vicreg-synth1b1-pqmfs")

for artifact_type in project.artifacts_types():
    if artifact_type.type != "model":
        continue
    collection_versions = []
    for artifact_collection in tqdm(artifact_type.collections()):
        for version in artifact_collection.versions():
            if version.state != "DELETED":
                collection_versions.append((artifact_collection, version))

for (artifact_collection, version) in tqdm(collection_versions):
    if len(version.aliases) > 0:
        # print out the name of the one we are keeping
        print(f"KEEPING {version.name} {version.aliases}")
    else:
        if not dry_run:
            version.delete()
        else:
            print("")
            print(f"should delete {version.name}")
