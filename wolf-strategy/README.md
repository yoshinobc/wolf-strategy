Aiwolf-strategies with personality
====

## Agents that use different strategies depending on their personality.

## Description
There are four types of agents here.  

1.Repel Agent  
自分に対して敵対的行動を起こしたエージェントを狙う．
repelTargetQue: 自分に対して投票すると宣言したエージェント，自分を人狼，裏切り者だと判断したエージェントを追加，追加されているエージェントが死んだ時はqueから除外  
AGREESentenceQue: 現在の自分の狙い相手に対して人狼，狂人であるとESTIMATEしている，または投票発言をしている場合に追加
DISAGREEQue: 現在の自分の狙い相手に対して村人，占い師であるとESTIMATEしているエージェント
	* vote:  自分に対して投票したエージェント，自分を人狼，裏切り者だと判定したエージェント
	* talk:  村人CO(占い師，裏切り者の場合は占いCO)
                 自分の投票先を素直に申告．まだ，投票先が定まっていない場合は自分以外のエージェントをランダムに申告
                 Because文で投票先が人狼なので投票したと発言
                 Agreequeに入っているtalk内容に対してAgree発言
                 DisAgreequeに入っているtalk内容に対してDisAgree発言
                 周りの人に投票を依頼
                 自分の敵対相手がWEREWOLFであると発言

2. Calm Agent
自分の戦略である疑い度合いに従って行動を決定する.  
suspicion: Human  
           人狼だと疑われた+2
           人狼だと疑った+1
           村人だと疑われた-1
           自分の役職を当てられた+10
           人狼だとCOした+10
           占い師だとCOした+2
           占いの結果人狼，狂人だと判明+100
           Wolf  
           人狼だと疑われた-2
           人狼だと疑った+1
           村人だと疑われた+1
           自分が人狼だと思われた+5
           自分が村人だと思われた-5  
	* vote: 疑い度合いが一番高いエージェントに投票
	* talk: 村人だとCO,(占い師，狂人の場合は占い師CO),一番疑い度合いが多いエージェントに投票，投票先を変更するときは自分のESTIMATE発言にDISAGREE,because文で自分の投票先が人狼であるとESTIMATEしたから投票と発言，周りに投票するように訴えかける，発言のAGREE,DISAGREE,まだCOしていないエージェントがいた時にCOするようにREQUEST
