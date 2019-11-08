declare -A Agents;

Agents=(
    [0]="Follow.py"
    [1]="Repel.py"
    [2]="Calm.py"
    [3]="Liar.py"
) #接続するエージェント
max=10000 #実行するAutoStarter.shの回数(ゲームの実行回数はmax*100回)
for ((i=0;i<$max;i++));do
    num1=`expr $RANDOM % 4`
    num2=`expr $RANDOM % 4`
    num3=`expr $RANDOM % 4`
    num4=`expr $RANDOM % 4`
    bash AutoStarter.sh &
    sleep 4 ; python3 ../${Agents[$num1]} -h localhost -p 10000 -n 1 &
    sleep 4 ; python3 ../${Agents[$num2]} -h localhost -p 10000 -n 2 &
    sleep 4 ; python3 ../${Agents[$num3]} -h localhost -p 10000 -n 3 & 
    sleep 4 ; python3 ../${Agents[$num4]} -h localhost -p 10000 -n 4
    if [ -e log/$i ]; then
        :
    else
        mkdir log/$i
    fi
    mv log/*.log log/$i/
done
