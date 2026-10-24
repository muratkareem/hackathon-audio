const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {JSDOM} = require('jsdom');
const root = path.resolve(__dirname,'..');
const translations = JSON.parse(fs.readFileSync(path.join(root,'i18n.json'),'utf8'));
const fixture = {
  id:'00000000-0000-4000-8000-000000000001', title:'Synthetic meeting / Учебная встреча', status:'complete',stage:'done',progress:100,
  summary_language:'en',speaker_names:{},warnings:[],diarization_status:'disabled',
  raw_data:[{id:0,start:0,end:3,text:'Anna will test the backend Friday.',speaker:null},{id:1,start:3,end:5,text:'Use SQLite for storage.',speaker:null}],
  report:{overview:'Synthetic report',topics:['Backend'],discussion:[],decisions:[{text:'Use SQLite',references:[1]}],
    actions:[{text:'Test backend',owner:'Anna',deadline:'Friday',references:[0]}],open_questions:[],takeaways:['Review tests']}
};
async function until(predicate){for(let i=0;i<30;i++){if(predicate())return;await new Promise(setImmediate);}assert.fail('UI did not settle');}
async function boot(t,options={}){
  const dom = new JSDOM(fs.readFileSync(path.join(root,'index.html'),'utf8'),{url:'http://localhost:8000/',runScripts:'outside-only'});
  t.after(async()=>{await new Promise(setImmediate);dom.window.close();});
  const w=dom.window,d=w.document,calls=[],downloads=[],blobs=[];
  w.HTMLElement.prototype.scrollIntoView=function(){};
  w.URL.createObjectURL=blob=>{blobs.push(blob);return 'blob:synthetic';};
  w.URL.revokeObjectURL=()=>{};
  w.HTMLAnchorElement.prototype.click=function(){downloads.push({connected:this.isConnected,name:this.download,href:this.href});};
  w.confirm=()=>false;
  Object.defineProperty(w.navigator,'clipboard',{value:{writeText:async text=>{w.copied=text;}}});
  const meeting=structuredClone(fixture);
  if(options.unsafeTitle)meeting.title='<script>alert(1)</script> Учебная встреча';
  w.fetch=async(url,request={})=>{
    calls.push({url,request});
    let value;
    if(url==='i18n.json')value=translations;
    else if(url==='/api/health')value={ollama:'ready',llm_model:'qwen2.5:7b',whisper_model:'small',device:'cpu',compute_type:'int8',ffmpeg:true,max_upload_bytes:200*1048576,retain_audio:false};
    else if(url==='/api/meetings')value=[meeting];
    else if(url.endsWith('/chat'))value=options.answer?await options.answer():{answer:'Anna',references:[0],insufficient_evidence:false};
    else if(request.method==='DELETE')return {ok:true,status:204};
    else value=meeting;
    return {ok:true,status:200,json:async()=>structuredClone(value),blob:async()=>new w.Blob(['Synthetic export'],{type:'text/plain'})};
  };
  w.eval(fs.readFileSync(path.join(root,'app.js'),'utf8'));
  await until(()=>d.querySelector('#dashboardMeetings button')&&d.querySelector('#health').textContent.includes('qwen2.5:7b'));
  return {w,d,calls,downloads,blobs,open:async()=>{d.querySelector('#dashboardMeetings button').click();await until(()=>!d.querySelector('#meeting').hidden);}};
}
test('navigation, settings, and independent persistent interface language',async t=>{
  const {w,d}=await boot(t);
  d.querySelector('[data-view="settings"]').click();
  assert.equal(d.querySelector('#settingsPage').hidden,false);
  assert.match(d.querySelector('#health').textContent,/CPU \/ int8/);
  d.querySelector('#locale').value='ru';d.querySelector('#locale').dispatchEvent(new w.Event('change'));
  assert.equal(w.localStorage.getItem('steppe-language'),'ru');
  assert.equal(d.documentElement.lang,'ru');
  assert.equal(d.querySelector('#summaryLanguage').value,'en');
  assert.equal(d.querySelector('#audioLanguage').value,'auto');
  d.querySelector('[data-view="upload"]').click();assert.equal(d.querySelector('#uploadPage').hidden,false);
  assert.equal(d.querySelector('#settingsPage').hidden,true);
});
test('action table references, transcript search, and copy use actual meeting data',async t=>{
  const {w,d,open}=await boot(t);await open();
  d.querySelector('[data-tab="actions"]').click();
  assert.match(d.querySelector('#actions').textContent,/Anna/);
  assert.match(d.querySelector('#actions').textContent,/Friday/);
  d.querySelector('#actions .reference').click();
  assert.equal(d.querySelector('#transcript').hidden,false);
  assert.equal(d.querySelector('#segment-0').classList.contains('match'),true);
  d.querySelector('#search').value='SQLite';d.querySelector('#search').dispatchEvent(new w.Event('input'));
  assert.equal(d.querySelector('#segment-0').hidden,true);assert.equal(d.querySelector('#segment-1').hidden,false);
  d.querySelector('#copy').click();await until(()=>w.copied);
  assert.match(w.copied,/\[00:00:00\].*Anna/);
});
test('mind map safely renders Cyrillic, zooms, and creates a valid attached SVG download',async t=>{
  const {w,d,open,downloads,blobs}=await boot(t,{unsafeTitle:true});await open();
  d.querySelector('[data-tab="map"]').click();
  const before=Number(d.querySelector('#mapCanvas svg').getAttribute('width'));
  d.querySelector('#zoomIn').click();assert.ok(Number(d.querySelector('#mapCanvas svg').getAttribute('width'))>before);
  d.querySelector('#svg').click();
  assert.deepEqual(downloads[0],{connected:true,name:'meeting-map.svg',href:'blob:synthetic'});
  const content=await new Promise((resolve,reject)=>{const reader=new w.FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=reject;reader.readAsText(blobs[0]);});
  const svg=new w.DOMParser().parseFromString(content,'image/svg+xml');
  assert.equal(svg.querySelector('parsererror'),null,svg.querySelector('parsererror')?.textContent + content.slice(0,300));assert.equal(svg.querySelector('script'),null);
  assert.match(svg.documentElement.textContent,/Учебная встреча/);
  assert.equal(svg.documentElement.getAttribute('xmlns'),'http://www.w3.org/2000/svg');
});
test('chat shows pending state, escapes answer text, and retains source navigation',async t=>{
  let answer;
  const {d,open}=await boot(t,{answer:()=>new Promise(resolve=>{answer=resolve;})});await open();
  d.querySelector('[data-tab="chat"]').click();d.querySelector('#question').value='Who tests the backend?';d.querySelector('#ask').click();
  await until(()=>answer);
  assert.equal(d.querySelector('#ask').disabled,true);assert.equal(d.querySelector('#messages').getAttribute('aria-busy'),'true');
  answer({answer:'Anna <script>alert(1)</script>',references:[0],insufficient_evidence:false});
  await until(()=>!d.querySelector('#ask').disabled);
  assert.equal(d.querySelector('#messages script'),null);assert.match(d.querySelector('#messages').textContent,/Anna/);
  d.querySelector('#messages .reference').click();assert.equal(d.querySelector('#transcript').hidden,false);
});
test('meeting deletion requires confirmation and returns to the library',async t=>{
  const {w,d,open,calls}=await boot(t);await open();d.querySelector('#delete').click();
  assert.equal(calls.some(c=>c.request.method==='DELETE'),false);
  w.confirm=()=>true;d.querySelector('#delete').click();
  await until(()=>d.querySelector('#meeting').hidden);
  assert.equal(calls.filter(c=>c.request.method==='DELETE').length,1);assert.equal(d.querySelector('#dashboardPage').hidden,false);
});
