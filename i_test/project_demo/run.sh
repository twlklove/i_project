values=$*
for value in ${values[@]} ; do printf \\x`printf "%x" $value`; echo -n " "; done; echo
