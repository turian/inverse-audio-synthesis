#!/usr/bin/env python3
# https://community.wandb.ai/t/using-the-python-api-to-delete-models-with-no-tag-minimal/1498?u=turian
# "Rather than using api.artifact_versions, it uses the versions
# method on artifact_collection."

import wandb

dry_run = True
#api = wandb.Api(overrides={"project": "vicreg-synth1b1-pqmfs", "entity": "turian"})
api = wandb.Api()
project = api.project('vicreg-synth1b1-pqmfs')

for artifact_type in project.artifacts_types():
    for artifact_collection in artifact_type.collections():        
        for version in artifact_collection.versions():
            if artifact_type.type == 'model':
                if len(version.aliases) > 0:
                    # print out the name of the one we are keeping
                    print(f'KEEPING {version.name} {(version.aliases, artifact_collection.name, artifact_type.type)}')
                else:
                    print(f'DELETING {version.name}')
                    if not dry_run:
                        print('')
                        version.delete()


