import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const workspaceDir=process.cwd();
const buildDir=path.join(workspaceDir,'build','deck');
const finalPath=process.env.FINAL_PPTX||path.join(workspaceDir,'output','Drug_Review_Defense.pptx');
const SKILL_DIR=process.env.PRESENTATION_SKILL_DIR;
const RUNTIME_PYTHON=process.env.RUNTIME_PYTHON;
if(!SKILL_DIR||!RUNTIME_PYTHON)throw new Error('Set PRESENTATION_SKILL_DIR and RUNTIME_PYTHON to the bundled Codex presentation runtime.');
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(SKILL_DIR,'container_tools/artifact_tool_utils.mjs')).href);
const input=JSON.parse(await fs.readFile(path.join(buildDir,'presentation_data.json'),'utf8'));
const presentation=Presentation.create({slideSize:{width:1280,height:720}});
const font='Arial';
const ink='#172E46';
await fs.mkdir(path.join(buildDir,'previews'),{recursive:true});

function text(slide,value,x,y,w,h,size=28,bold=false,color=ink){
  const shape=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  shape.text=String(value);
  shape.text.style={typeface:font,fontSize:size,bold,color,verticalAlignment:'middle',wrap:'square'};
  return shape;
}
function table(slide,headers,rows,y=190,h=270,widths){
  const values=[headers,...rows].map(row=>row.map(String));
  const t=slide.tables.add({rows:values.length,columns:headers.length,left:72,top:y,width:1136,height:h,values,
    ...(widths?{columnWidths:widths}:{})});
  t.cells.block({row:0,column:0,rowCount:values.length,columnCount:headers.length}).assign({
    fill:'#FFFFFF',textStyle:{typeface:font,fontSize:24,color:ink},margins:{left:12,right:12,top:10,bottom:10}});
  t.cells.block({row:0,column:0,rowCount:1,columnCount:headers.length}).assign({fill:'#E8EFF6',textStyle:{typeface:font,fontSize:23,bold:true,color:ink}});
  t.borders.assign({fill:'#CAD5DF',width:1});
  return t;
}
function countChart(slide,title,categories,values,x,w){
  const chart=slide.charts.add('bar',{
    position:{left:x,top:180,width:w,height:340},title,titleTextStyle:{fontSize:25,fill:ink,bold:true},
    categories,series:[{name:'Training reviews',values,fill:'#2563A6'}],hasLegend:false,
    barOptions:{direction:'column',grouping:'clustered',gapWidth:75},
    xAxis:{textStyle:{fontSize:19,fill:ink},line:{fill:'#8090A1',width:1}},
    yAxis:{min:0,max:Math.ceil(Math.max(...values)*1.25/1000)*1000,numberFormatCode:'#,##0',
      textStyle:{fontSize:19,fill:ink},majorGridlines:{fill:'#D9E0E7',width:1}},
    dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'#,##0',textStyle:{fontSize:19,fill:ink}},
    chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF'
  });
  applyPresentationChartFont(chart,{fontFamily:font});
}
for(let i=0;i<input.slides.length;i++){
  const s=input.slides[i];
  const slide=presentation.slides.add();slide.background.fill='#FFFFFF';
  text(slide,s.title,72,60,1136,80,i===0?56:44,true);
  text(slide,`${i+1} / ${input.slides.length}`,1110,666,100,28,17,false,'#52677D');
  if(i===0){
    text(slide,s.subtitle,72,210,1100,100,34,false,'#2563A6');
    s.lines.forEach((line,j)=>text(slide,line,72,390+j*54,1100,46,28));
  }else if(i===2){
    countChart(slide,'Training class counts',input.classCategories,input.classCounts,72,530);
    countChart(slide,'Review length in tokens',input.lengthCategories,input.lengthCounts,660,548);
    s.lines.forEach((line,j)=>text(slide,line,72,550+j*44,1136,42,25));
  }else if(i===8){
    for(let j=0;j<3;j++){
      const block=input.confusions[j];const left=72+j*384;
      text(slide,block.name+' selected',left,165,350,38,26,true);
      const values=[['True','Neg','Neu','Pos'],...block.counts.map((r,k)=>[["Neg","Neu","Pos"][k],...r.map(String)])];
      const t=slide.tables.add({rows:4,columns:4,left,top:220,width:350,height:220,values,columnWidths:[98,84,84,84]});
      t.cells.block({row:0,column:0,rowCount:4,columnCount:4}).assign({fill:'#FFFFFF',textStyle:{typeface:font,fontSize:23,color:ink},margins:{left:7,right:7,top:8,bottom:8}});
      t.cells.block({row:0,column:0,rowCount:1,columnCount:4}).assign({fill:'#E8EFF6',textStyle:{typeface:font,fontSize:23,bold:true,color:ink}});
      t.borders.assign({fill:'#CAD5DF',width:1});
    }
    text(slide,'Rows: true class. Columns: predicted class.',72,452,1136,32,21);
    s.lines.forEach((line,j)=>text(slide,line,72,500+j*45,1136,42,25));
  }else if(s.rows){
    const heights=i===6?280:260;
    table(slide,s.headers,s.rows,185,heights,i===7?[150,230,190,350,216]:undefined);
    s.lines.forEach((line,j)=>text(slide,line,72,505+j*44,1136,42,25));
  }else{
    s.lines.forEach((line,j)=>text(slide,line,72,195+j*76,1136,67,30));
  }
  slide.speakerNotes.text=`Planned time: ${s.seconds} seconds\n\n${s.notes}\n\nSources: Course Guidelines_RNN_2627.pdf, supplied Drugs.com CSVs, data/processed/data_audit.json, artifacts/validation_results.csv, artifacts/test_results.csv.\nGreff et al.: https://arxiv.org/abs/1503.04069\nLandi et al.: https://arxiv.org/abs/2109.00020\n${input.repoUrl}`;
}
const candidatePath=path.join(buildDir,'candidate.pptx');
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
for(const [i,slide] of presentation.slides.items.entries()){
  const blob=await presentation.export({slide,format:'png',scale:2});
  await fs.writeFile(path.join(buildDir,'previews',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const result=await finalizePresentation({
  workspaceDir,candidatePath,finalPath,
  pythonExecutable:RUNTIME_PYTHON,
  integrityValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',
    ...[6,7,8,9].flatMap(n=>['--require-native-table-slide',String(n)])],
  explicitTotalSlideCount:10,requiredNativeTableOwnerSlides:[6,7,8,9],requiredNativeChartOwnerSlides:[3],
  materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,
  receiptPath:path.join(buildDir,'validation.json')
});
console.log(JSON.stringify({finalPath:result.finalPath??'output/Drug_Review_Defense.pptx'}));
