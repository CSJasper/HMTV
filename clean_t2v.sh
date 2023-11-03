cd ./text2video
cp ./make_video.py ../make_video.py
cp -r ./motion2skeleton ../
rm -rf *
mv ../make_video.py ./
mv ../motion2skeleton ./
cd ../
cd ./text2motion
cp ./text2motion.py ../text2motion.py
rm -rf *
mv ../text2motion.py ./
cd ../