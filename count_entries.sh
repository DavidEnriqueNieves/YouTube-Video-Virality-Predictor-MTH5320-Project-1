count=0
for i in ./channels/*;

do 
    
    { # try
        echo `cat ${i}/videos.txt |wc -w`
        count=$((count+`cat ${i}/videos.txt |wc -w`));
    } || { # catch
        # save log for exception 
        echo "lol"
    }
done
echo $count