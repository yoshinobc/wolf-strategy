declare -A Agents;

Agents=(
    [0]="yskn67/run.py"
    [1]="sonoda/python_sample.py"
    [2]="cantar/cantar_player.py"
    [3]="Litt1eGirl/Litt1eGirlPlayer.py"
    [4]="calups/agent.py"
) #接続するエージェント
max=2000 #実行するAutoStarter.shの回数(ゲームの実行回数はmax*100回)
for ((i=0;i<$max;i++));do
    num1=`expr $RANDOM % 5`
    num2=`expr $RANDOM % 5`
    num3=`expr $RANDOM % 5`
    num4=`expr $RANDOM % 5`
    num5=`expr $RANDOM % 5`
    bash AutoStarter.sh &
    sleep 4 ; python3 ../../../Downloads/${Agents[$num1]} -h localhost -p 10000 -r 1 &
    sleep 4 ; python3 ../../../Downloads/${Agents[$num2]} -h localhost -p 10000 -r 2 &
    sleep 4 ; python3 ../../../Downloads/${Agents[$num3]} -h localhost -p 10000 -r 3 & 
    sleep 4 ; python3 ../../../Downloads/${Agents[$num4]} -h localhost -p 10000 -r 4 &
    sleep 4 ; python3 ../../../Downloads/${Agents[$num5]} -h localhost -p 10000 -r 5
    if [ -e log/$i ]; then
        :
    else
        mkdir log/$i
    fi
    mv log/*.log log/$i/
done
