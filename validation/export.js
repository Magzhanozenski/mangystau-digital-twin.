// Экспорт данных сети и результатов JS-решателя для независимой проверки
const E=require('../core_js/engine2.js'); const fs=require('fs');
const rnd=(()=>{let a=12345;return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};})();
const cases=[];
for(let k=0;k<400;k++){
  const T=-30+rnd()*78, load=90+rnd()*170;
  const ids=E.LINES.map(l=>l.id); const nOut=[0,1,1,2,2,3][Math.floor(rnd()*6)]; const linesOut=[];
  while(linesOut.length<nOut){const id=ids[Math.floor(rnd()*ids.length)];if(!linesOut.includes(id))linesOut.push(id);}
  const gensOut=[]; if(rnd()<0.15)gensOut.push('G2'); if(rnd()<0.15)gensOut.push('G3');
  const ctrl={gP:{G2:Math.round(rnd()*50),G3:Math.round(rnd()*28)},vset:{G1:1.0+rnd()*0.07,G2:1.0+rnd()*0.06,G3:1.0+rnd()*0.06},bsk:{4:rnd()<0.3?10:0,8:rnd()<0.3?5:0},shed:{}};
  const r=E.runPowerFlow({T,load,linesOut,gensOut,ctrl});
  if(r.status!=='ok') { cases.push({T,load,linesOut,gensOut,ctrl,status:r.status,lambdaMax:r.lambdaMax,islands:r.islands,loadBus:r.loadBus,gen:Object.fromEntries(Object.entries(r.gen).map(([k,g])=>[k,{...g,atQlim:false}])),pvGensQlim:r.pv&&r.pv.sol?null:null}); continue; }
  cases.push({T,load,linesOut,gensOut,ctrl,status:r.status,V:r.V,ang:r.ang,flows:r.flows,gen:r.gen,islands:r.islands,loadBus:r.loadBus,achr:r.achr});
}
fs.writeFileSync('validation/cases.json',JSON.stringify({BUSES:E.BUSES,LINES:E.LINES,GENS:E.GENS,COND:E.COND,cases}));
// Якобиан: аналитический vs конечные разности
const s={T:35,load:180,linesOut:[],gensOut:[],ctrl:E.defaultCtrl()};
const ids=E.BUSES.map(b=>b.id); const isl=E.buildIsland(ids,{T:35},E.LINES);
const V=ids.map((_,i)=>1-0.02*i), th=ids.map((_,i)=>-0.03*i);
const pvq=[1,2,3,4,5,6,7], pq=[1,2,3,6,7];
const {P,Q}=E.calcPQ(isl,V,th); const J=E.buildJ(isl,pvq,pq,V,th,P,Q);
const f=(V,th)=>{const r=E.calcPQ(isl,V,th);return [...pvq.map(i=>r.P[i]),...pq.map(i=>r.Q[i])];};
let maxErr=0,maxVal=0;const h=1e-7;const f0=f(V,th);
[...pvq.map(i=>['th',i]),...pq.map(i=>['V',i])].forEach(([kind,i],c)=>{const V2=V.slice(),t2=th.slice();if(kind==='th')t2[i]+=h;else V2[i]+=h;const f1=f(V2,t2);f1.forEach((v,r)=>{const num=(v-f0[r])/h;maxErr=Math.max(maxErr,Math.abs(num-J[r][c]));maxVal=Math.max(maxVal,Math.abs(J[r][c]));});});
console.log('Якобиан: макс. расхождение с конечными разностями',maxErr.toExponential(2),'при макс. элементе',maxVal.toFixed(1));
console.log('Случаев:',cases.length,'сошлось:',cases.filter(c=>c.status==='ok').length,'pv_limit:',cases.filter(c=>c.status==='pv_limit').length,'numerical:',cases.filter(c=>c.status==='numerical').length);
