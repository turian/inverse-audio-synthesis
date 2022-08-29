import datetime

import git


def utcstr():
    return datetime.datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")


utcnowstr = utcstr()

git_repo = git.Repo(search_parent_directories=True)
git_sha = git_repo.head.object.hexsha
