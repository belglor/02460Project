# 02460 Project
Advanced Machine Learning project

By <Member name 1>, <Member name 2>, <Member name 3> and <Mirza Hasanbasic>

Table of Contents
=================

* [About this project](https://github.com/kazyka/02460Project#About-this-project)
    * [How to use](https://github.com/kazyka/02460Project#How-to-use)
* [How to use git](https://github.com/kazyka/02460Project#how-to-use-git)
    * [Before you start](https://github.com/kazyka/02460Project#before-you-start)
    * [Branching](https://github.com/kazyka/02460Project#)
    * [Git status and diff](https://github.com/kazyka/02460Project#Git-status-and-diff)
    * [Update and merge](https://github.com/kazyka/02460Project#Update-and-merge)
    * [Replace local changes](https://github.com/kazyka/02460Project#replace-local-changes)
    * [Commiting and pushing](https://github.com/kazyka/02460Project#commiting-and-pushing)
* [Python requirement](https://github.com/kazyka/02460Project#python-requirementtxt)


For Markdown [Click here](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)

# About this project

## How to use

# How to use git

[Link about branching](http://nvie.com/posts/a-successful-git-branching-model/)

The basic usage of git should be, that you need to clone the project, this is done `git clone <path to repo>` remember, that you can use https, then you need to type your passwordd and username everytime or you can create a ssh

## Before you start

You should set your details. To do this run the following in your terminal

```
$ git config --global user.name 'Your Name'
$ git config --global user.email you@somedomain.com
```

## Branching

A branch is not available to others unless you push the branch to your remote repository

First we create a new branch

```
$ git branch BRANCH_NAME    # Create a new branch
$ git checkout BRANCH_NAME  # Then switch to the new branch
```

## Git status and diff

During your owrk, you may wish to check on what files you've edited, removed and added. This can be done by checking the status of the repository

```
$ git status
```

If you wish to check the actual changes made to a file, this can be done with

```
$ git diff <filename>
```

before merging changes, you can also preview them by using

```
$ git diff <source_filename> <target_filename>
```

more about merge below.

## Update and merge

To update your local repository to the newest commit, execute

```
$ git pull
```
in your working directory to fetch and merge remote changes.
to merge another branch into your active branch (e.g. master), use

```
$ git merge <branch>
```

in both cases git tries to auto-merge changes. Unfortunately, this is not always possible and results in conflicts. You are responsible to merge those conflicts manually by editing the files shown by git. After changing, you need to mark them as merged with

```
$ git add <filename>
```

before merging changes, you can also preview them by using

```
$ git diff <source_filename> <target_filename>
```

Rather than manually specify each file, you can run

```
$ git add assets/css
```

This will add the entire css folder. This can be even more simplified with `.`

```
$ git add .
```

To add a file to your HEAD which has been removed (deleted) from the file structure, git add won’t do, instead you need to run git rm.

```
$ git rm <filename>
```

If this file still exists - this command will delete the file and add the deletion to the staging area (two birds, one stone).

There are a couple of flags you can add to the your git add command to help the process a little easier.

```
$ git add -u
```

This command will add modified and deleted files to the staging area - but not new ones. Handy if you have lots of new files but want to commit them separately to newly created files.

```
$ git add -A
```

The -A flag is the daddy of all flags - running this command will add modified, deleted and new files to the staging area. Especially handy if you are running a first commit on a feature branch or tidying up files.

With both of these commands, the folder can be specified after the flag to narrow down the blanket adding tp a specific location. This may be the case if you have some modified javascript, images and CSS, for example, and only wish to add the modified files in the CSS folder:

```
$ git add -u assets/css
```

There may be the odd occasion where you wish to remove a file from the staging area (or HEAD) or empty the staging area (without undoing your changes). It is also sometimes easier to add all the files (with the -u flag for example) and then unstage a particular file or folder.

To do this, you need to reset the staging area. To remove a specific file from the staging area (or HEAD) the command is:


```
$ git reset HEAD <filepath>
```

The filepath can be omitted to completely remove everything from the staging area.

```
$ git reset HEAD
```

## Replace local changes

In case you did someting wrong, you can replace local changes

```
$ git checkout -- <filename>
```

If you instead want to drop all your local changes and commits, fetch the latest history from the server and point your local master branch at it like this

```
git fetch origin
git reset --hard origin/master
```

## Commiting and pushing

```
$ git commit -m "text"
$ git push
```

```
$ git log       # Show commit logs
```

# Python Requirement.txt

```
pip install SomePackage              # latest version
pip install SomePackage==1.0.4       # specific version
pip install 'SomePackage>=1.0.4'     # minimum version
```

```
pip install -r requirements.txt
```
