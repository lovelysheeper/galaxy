# galaxy classify
frame:pytorch    
model:vgg16    
dataset:galaxy zoo challenge    
classes:"edge_on_disk" "elliptical" "face_on_disk" "merging"    

run "train.py" to train model    
put galaxy image in "predict" to predict class    

class1:t00 椭圆，盘星系，星或其他    
class2:t01 侧向盘，正向盘    
class3:t02 有棒， 无棒    
class4:t03 有旋臂， 无旋臂    
class5:t04 无核，弱核，强核，核支配 （正面）    
class6:t05 有异样，无异样     
class7:t07 很圆，略扁，雪茄状 （椭圆星系）    
class8:t06 环状，透镜，分裂，不规则，其他，并合，尘埃带 （异样）    
class9:t08 圆核， 花生核， 无核  （侧面）    
class10:t09 紧密，一般，松弛 （旋臂缠绕）    
class11:t10 一条，两条，三条，四条，更多，不清楚 （旋臂数量）    

椭圆星系：class1.1 > 0.7 and class6.2 > 0.7    
正向盘星系：class1.2 > 0.7 and class2.2/(class1.2 + 0.0001) > 0.7 and class6.2 > 0.7     
侧向盘星系：class1.2 > 0.7 and class2.1/(class1.2 + 0.0001) > 0.7 and class6.2 > 0.7    
并合星系：class 6.1 >0.7 and class8.6/(class6.2 + 0.0001) > 0.7    
