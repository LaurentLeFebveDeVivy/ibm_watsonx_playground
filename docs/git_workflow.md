# Git Workflow Guide

This document describes how we use git and GitHub on this project. Follow it step by step — every command is copy-pasteable.

**Table of Contents**

1. [Branch Structure](#1-branch-structure)
2. [One-Time Setup](#2-one-time-setup)
3. [Naming Conventions](#3-naming-conventions)
4. [Day-to-Day Workflow](#4-day-to-day-workflow)
5. [Merging dev into main](#5-merging-dev-into-main)
6. [Keeping Your Branch Up to Date](#6-keeping-your-branch-up-to-date)
7. [Handling Merge Conflicts](#7-handling-merge-conflicts)
8. [Common Mistakes and How to Fix Them](#8-common-mistakes-and-how-to-fix-them)
9. [Quick Reference Cheat Sheet](#9-quick-reference-cheat-sheet)
10. [Visual Workflow Summary](#10-visual-workflow-summary)

---

## 1. Branch Structure

```
main              ← stable, production-ready code
  └── dev         ← integration branch, where features come together
       ├── feature/add-auth
       ├── feature/cos-upload
       ├── fix/retrieval-timeout
       └── fix/config-validation
```

| Branch | Purpose | Who creates it | Who merges into it | Lifetime |
|---|---|---|---|---|
| `main` | Stable, production-ready code | Already exists | Project lead merges from `dev` | Permanent |
| `dev` | Integration branch — all work merges here first | Created once by project lead | PR authors (after review) | Permanent |
| `feature/*` | New functionality | Any developer | PR author merges into `dev` | Deleted after merge |
| `fix/*` | Bug fixes | Any developer | PR author merges into `dev` | Deleted after merge |

**Rules:**

- Never commit directly to `main` or `dev`. Always use a feature/fix branch and a pull request.
- `feature/*` and `fix/*` branches are always created from `dev` (not from `main`).
- `dev` is merged into `main` only when the team agrees the code is ready. The project lead does this.

---

## 2. One-Time Setup

### 2.1 Clone the repo

```bash
git clone git@github.com:LaurentLeFebveDeVivy/ibm_watsonx_playground.git
cd ibm_watsonx_playground
```

Then switch to the `dev` branch:

```bash
git checkout dev
git pull origin dev
```

### 2.2 Create the dev branch (already done)

```bash
git checkout main
git pull origin main
git checkout -b dev
git push -u origin dev
```

### 2.3 Configure your git identity (each developer, once per machine)

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

---

## 3. Naming Conventions

### Branch names

| Pattern | When to use | Examples |
|---|---|---|
| `feature/<short-description>` | Adding new functionality | `feature/add-auth`, `feature/cos-upload`, `feature/chat-streaming` |
| `fix/<short-description>` | Fixing a bug | `fix/retrieval-timeout`, `fix/config-validation`, `fix/cors-headers` |

Rules:

- Lowercase only
- Use hyphens to separate words (not underscores, not spaces)
- Keep it short but descriptive (2-4 words)

### Commit messages

Format: `type: short description of what you did`

| Prefix | When to use | Example |
|---|---|---|
| `feat:` | You added something new | `feat: add COS file upload endpoint` |
| `fix:` | You fixed a bug | `fix: handle missing API key in config` |
| `docs:` | You changed documentation only | `docs: add RAG pipeline explanation` |


```
feat: add document ingestion from COS

Downloads PDF files from the configured COS bucket,
splits them into chunks, and indexes them into Milvus.
```

---

## 4. Day-to-Day Workflow

This is the process you follow every time you work on something.

### Step 1: Start from dev and make sure it is up to date

```bash
git switch dev
git pull origin dev
```

*This switches you to the dev branch and downloads any new changes your teammates have pushed.*

### Step 2: Create your working branch

For a new feature:

```bash
git checkout -b feature/your-feature-name
```

For a bug fix:

```bash
git checkout -b fix/your-fix-name
```

*This creates a new branch based on dev and switches to it. All your work happens here.*

### Step 3: Do your work and make commits

```bash
# Check what you changed
git status


# stage specific files
git add path/to/file1.py path/to/file2.py

# Alternatively, stage all changes made
git add . # Run from repo root

# Commit with a message
git commit -m "feat: add document chunking logic"
```

**Tips:**

- Commit early and often. Small commits are easier to review and easier to undo if something goes wrong.
- Use `git status` frequently. It tells you where you are and what has changed.
- Use `git add <file>` for specific files rather than `git add .` to avoid accidentally committing files you did not intend (like `.env` or `__pycache__/`).

### Step 4: Push your branch to GitHub

The first time you push a new branch:

```bash
git push -u origin feature/your-feature-name
```

*The `-u` flag sets up tracking so future pushes only need `git push`.*

After the first push, just use:

```bash
git push
```

### Step 5: Create a Pull Request on GitHub

1. Go to the repository on GitHub.
2. GitHub will show a banner "Compare & pull request" for your recently pushed branch. Click it.
3. **Set the base branch to `dev`** (not `main`). This is important.
4. Write a title that describes what the PR does.
5. In the description, explain what you changed and why. If relevant, note how to test it.
6. Assign a teammate as reviewer.
7. Click "Create pull request".


### Step 6: Code review

- The reviewer reads the code on GitHub and either leaves comments or approves.
- If changes are requested, make them on your branch locally, commit, and push. The PR updates automatically.
- One approval is sufficient to merge.

### Step 7: Merge the PR

After approval:

1. The PR author clicks **"Merge pull request"** on GitHub. Use the default merge commit — do not select squash or rebase.
2. Click **"Delete branch"** when prompted. This cleans up the remote branch.

### Step 8: Clean up locally

```bash
git checkout dev
git pull origin dev
git branch -d feature/your-feature-name
```

*This switches you back to dev, pulls the merged changes, and deletes your local branch since the work is done.*

---

## 5. Merging dev into main

This is done by the **project lead only**, at agreed milestones (not after every feature).

```bash
git checkout main
git pull origin main
git merge dev
git push origin main
```

Alternatively, create a PR from `dev` into `main` on GitHub and merge it there.

---

## 6. Keeping Your Branch Up to Date

When you are working on a feature branch and `dev` has been updated by teammates, bring those changes into your branch:

```bash
git checkout dev
git pull origin dev
git checkout feature/your-feature-name
git merge dev
```

*This brings the latest dev changes into your branch.*

If there are no conflicts, git will open an editor for a merge commit message. Accept the default and save (in the terminal, this is usually `:wq` in vim, or just close the editor window).

**Do this regularly** — at least once a week when teammates are actively merging. Small, frequent merges produce small, easy conflicts. Large, infrequent merges produce large, painful conflicts.

---

## 7. Handling Merge Conflicts

A conflict happens when two people changed the same lines in the same file. Git cannot decide which version to keep, so it asks you.

### What a conflict looks like

When `git merge dev` reports a conflict, the affected file will contain markers like this:

```
<<<<<<< HEAD
your version of the code
=======
the other version of the code
>>>>>>> dev
```

### How to resolve it

1. Run `git status` to see which files are conflicted (listed as "both modified").

2. **Open the file in VS Code.** VS Code highlights the conflict and shows buttons:
   - **Accept Current Change** — keep your version
   - **Accept Incoming Change** — keep the other version
   - **Accept Both Changes** — keep both

   Pick the right option, or manually edit the code to combine both changes correctly.

3. After resolving all conflicts in all files:

```bash
git add path/to/resolved-file.py
git commit -m "fix: resolve merge conflict with dev"
```

### If you get overwhelmed

You can abort the merge and go back to where you were before:

```bash
git merge --abort
```

Then ask a teammate for help.

---

## 8. Common Mistakes and How to Fix Them

### 8.1 "I committed to the wrong branch"

If you have **not pushed yet**:

```bash
# Undo the last commit but keep the changes
git reset --soft HEAD~1

# Create the correct branch and switch to it
git checkout -b feature/correct-branch-name

# Your changes are still staged, so just commit again
git commit -m "feat: your message"
```

### 8.2 "I forgot to pull before starting work and now I cannot push"

```bash
git pull origin dev
```

If this causes a merge conflict, resolve it (see [Section 7](#7-handling-merge-conflicts)), then push.

### 8.3 "I want to undo my last commit (not pushed yet)"

```bash
git reset --soft HEAD~1
```

*This undoes the commit but keeps your changes staged, so you can edit and recommit.*

### 8.4 "I accidentally staged a file I do not want to commit"

```bash
git restore --staged path/to/file
```

### 8.5 "I want to see what changed before I commit"

```bash
# See unstaged changes
git diff

# See staged changes (what will be in the next commit)
git diff --staged
```

### 8.6 "I need to switch branches but I have uncommitted changes"

```bash
# Save your changes temporarily
git stash

# Switch to the other branch
git checkout dev

# ... do what you need to do ...

# Switch back and restore your changes
git checkout feature/your-feature-name
git stash pop
```

### 8.7 "I pushed something I should not have"

Do not rewrite history. Instead, create a new commit that undoes the mistake:

```bash
git revert HEAD
git push
```

*This creates a new commit that reverses the previous one. It is safe because it does not rewrite history.*

### 8.8 "I am completely lost"

```bash
git status
git log --oneline -10
```

*These two commands tell you which branch you are on, what files are changed, and what recent commits look like. If you are still stuck, ask a teammate.*

---

## 9. Quick Reference Cheat Sheet

| I want to... | Command |
|---|---|
| See which branch I am on | `git status` |
| Switch to dev and update it | `git checkout dev && git pull origin dev` |
| Create a new feature branch | `git checkout -b feature/name` |
| Create a new fix branch | `git checkout -b fix/name` |
| See what I changed | `git diff` |
| See what is staged | `git diff --staged` |
| Stage a file | `git add path/to/file` |
| Stage all changed files | `git add .` |
| Commit | `git commit -m "feat: description"` |
| Push my branch (first time) | `git push -u origin feature/name` |
| Push my branch (after first time) | `git push` |
| Create a PR (CLI) | `gh pr create --base dev --title "Title" --body "Description"` |
| Update my branch with latest dev | `git checkout dev && git pull origin dev && git checkout feature/name && git merge dev` |
| Undo last commit (not pushed) | `git reset --soft HEAD~1` |
| Unstage a file | `git restore --staged path/to/file` |
| Save uncommitted work temporarily | `git stash` |
| Restore stashed work | `git stash pop` |
| Abort a failed merge | `git merge --abort` |
| Delete a local branch after merge | `git branch -d feature/name` |
| See recent commits | `git log --oneline -10` |
| Undo a pushed commit safely | `git revert HEAD && git push` |

---

## 10. Visual Workflow Summary

```
  dev (always start here)
   │
   ├── 1. git checkout -b feature/xyz
   │       │
   │       ├── 2. Work, commit, work, commit
   │       │
   │       ├── 3. git push -u origin feature/xyz
   │       │
   │       ├── 4. Open Pull Request (base: dev)
   │       │
   │       ├── 5. Teammate reviews code
   │       │
   │       └── 6. Merge PR into dev
   │               │
   │               └── 7. Delete feature/xyz branch
   │
   └── (at milestones) Project lead merges dev → main
```
