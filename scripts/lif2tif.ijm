macro 'lif2tif' {
run("Bio-Formats Macro Extensions");
d1 = getdectory("folder");
list = getFileList(d1);
setBatchMode(true);
	d1dir = File.getdir(d1);
	d1name = File.getName(d1);
	d2 = d1dir+File.separator+d1name+"--mip";
	if (File.exists(d2)==false) {
				File.makedectory(d2);
		}
for (i=0; i<list.length; i++) {
	print("processing ... "+i+1+"/"+list.length+"\n         "+list[i]);
	path=d1+list[i];
	Ext.setId(path);
	Ext.getSeriesCount(seriesCount); 
	for (j=1; j<=seriesCount; j++) {
		run("Bio-Formats", "open=path autoscale color_mode=Default view=Hyperstack stack_order=XYCZT series_"+j);
		name=File.nameWithoutExtension;
		text=getMetadata("Info");
		n1=indexOf(text," Name = ")+8;
		n2=indexOf(text,"SizeC = ");
		seriesname=substring(text, n1, n2-2);
		if (nSlices>1) {
			run("Z Project...", "projection=[Max Intensity]");
			if (nSlices>1) Stack.setDisplayMode("composite");
			saveAs("Tiff", d2+File.separator+name+"_"+seriesname+"_MIP.tif");
		}
		else saveAs("TIFF", d2+File.separator+name+"_"+seriesname+".tif");
		run("Close All");
		run("Collect Garbage");
	}
}
run("Close All");
setBatchMode(false);
} // macro
