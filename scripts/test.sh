timer_start=`date "+%Y-%m-%d %H:%M:%S"`
sleep 2s
timer_end=`date "+%Y-%m-%d %H:%M:%S"`
duration=`echo $(($(date +%s -d "${timer_end}") - $(date +%s -d "${timer_start}"))) | awk '{t=split("60 s 60 m 24 h 999 d",a);for(n=1;n<t;n+=2){if($1==0)break;s=$1%a[n]a[n+1]s;$1=int($1/a[n])}print s}'`

a="hello"
style="data/styles/$a.jpg"

echo "开始： $timer_start"
a="world"
echo $style
echo $style
echo "结束： $timer_end"
echo "耗时： $duration"