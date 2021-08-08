#clone a git project
git clone xxxxx

# don't input passwd everytime
git config --global credential.helper store  

# lookup branch and addr
git remote -v
git branch 

# creat branch, switch branch or del branch
git checkout -B i_branch_1 
git checkout i_branch_1
git branch -D i_branch_1

# update or download files
git update
git checkout .

# upload
git add xxx
git commit -m "xx"
git push

