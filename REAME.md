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

#
create a new repositorie named "i_project"  in web github

#
ssh-keygen -t rsa -C "12345678@qq.com"

#config
git config --global user.name "xxxx" 
git config --global user.email xxxxx@qq.com 
#create a new repository on the command line
echo "# i_project" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/twlklove/i_project.git
git push -u origin main

