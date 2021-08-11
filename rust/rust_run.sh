project=test0
cargo new $project 
cd $project
cargo check

#build
cargo build  #for debug

cargo build --release #or for release

#run
cargo run


#or use rustc build
#rustc main.rs

