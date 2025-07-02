wget https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip train.zip
rm train.zip
rm -rf __MACOSX
cd train
unzip train_databases.zip
rm train_databases.zip
rm -rf __MACOSX
cd ..
