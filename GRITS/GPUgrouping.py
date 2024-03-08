from bifrost import map as BFMap 
def labelgroups(islanddata,ngrid,ints,nchan):
    # LABELGROUPS
    BFMap("""
    auto counter = 0;
    for (auto i = 1; i < %i; i++) {
    	for (auto j = 1; j < %i; j++) {
    		if (0 != isldata(time,chan,stokes,i,j)) {
    			counter++;
    			isldata(time,chan,stokes,i,j) += counter - 1;
    			if (0 != isldata(time,chan,stokes,i,j - 1)){
                                isldata(time,chan,stokes,i,j) = isldata(time,chan,stokes,i,j - 1);
                                }
    		}
    	}
    }
        """ % (ngrid,ngrid,), 
        {'isldata':islanddata}, 
        axis_names=('time','chan','stokes'), shape=(ints,nchan,4))
        
def groupvertcols(data,grid):
    BFMap("""
    if ((j>0) && (j<%i-1)) {
    	for (auto i = 1; i < %i-1; i++) {
    		if ((0 != isldata(time,chan,stokes,i,j)) && (0 != isldata(time,chan,stokes,i - 1,j))) {
    			isldata(time,chan,stokes,i,j) = min(isldata(time,chan,stokes,i - 1,j), isldata(time,chan,stokes,i,j));
    		}
    	}
    	for (int i = %i - 1; i > 0; i--) {
    		if ((0 != isldata(time,chan,stokes,i,j)) && (0 != isldata(time,chan,stokes,i + 1,j))) {
    			isldata(time,chan,stokes,i,j) = min(isldata(time,chan,stokes,i + 1,j), isldata(time,chan,stokes,i,j));
    		}
    	}
     }
        """ % (grid,grid,grid,), 
        {'isldata':data}, 
        axis_names=('time','chan','stokes','j'), shape=(data.shape[0],data.shape[1],data.shape[2],data.shape[3]))
        
def grouphorizrows(data,grid):
    BFMap("""
        if ((i>0) && (i<%i-1)) {
    	for (auto j = 1; j < %i-1; j++) {
    		if ((0 != isldata(time,chan,stokes,i,j)) && (0 != isldata(time,chan,stokes,i,j + 1)) && (isldata(time,chan,stokes,i,j + 1) < isldata(time,chan,stokes,i,j))) {
    			for (auto k = 0; k < %i; k++) if (isldata(time,chan,stokes,i,k) == isldata(time,chan,stokes,i,j)) isldata(time,chan,stokes,i,k) = isldata(time,chan,stokes,i,j + 1);
    		}
    		if ((0 != isldata(time,chan,stokes,i,j)) && (0 != isldata(time,chan,stokes,i,j - 1)) && (isldata(time,chan,stokes,i,j - 1) < isldata(time,chan,stokes,i,j))) {
    			for (auto k = 0; k < %i; k++) if (isldata(time,chan,stokes,i,k) == isldata(time,chan,stokes,i,j)) isldata(time,chan,stokes,i,k) = isldata(time,chan,stokes,i,j - 1);
    		}
    	}
    }
        """ % (grid,grid,grid,grid,), 
        {'isldata':data}, 
        axis_names=('time','chan','stokes','i'), shape=(data.shape[0],data.shape[1],data.shape[2],data.shape[3]))

def groupdiagcols(data,grid):
    BFMap("""
    if ((j>1) && (j<%i-2)) {
    	for (auto i = 2; i < %i-2; i++) {
    		if ((0 != rm(time,chan,stokes,i,j)) && (0 != rm(time,chan,stokes,i - 2,j))) {
    			rm(time,chan,stokes,i,j) = min(rm(time,chan,stokes,i - 2,j), rm(time,chan,stokes,i,j));
    		}
    	}
    	for (int i = %i - 2; i > 1; i--) {
    		if ((0 != rm(time,chan,stokes,i,j)) && (0 != rm(time,chan,stokes,i + 2,j))) {
    			rm(time,chan,stokes,i,j) = min(rm(time,chan,stokes,i + 2,j), rm(time,chan,stokes,i,j));
    		}
    	}
     }
        """ % (grid,grid,grid,), 
        {'rm':data}, 
        axis_names=('time','chan','stokes','j'), shape=(data.shape[0],data.shape[1],data.shape[2],data.shape[3]))

def groupdiagrows(data, grid):
    BFMap("""
        if ((i>1) && (i<%i-2)) {
    	for (auto j = 2; j < %i-2; j++) {
    		if ((0 != rm(time,chan,stokes,i,j)) && (0 != rm(time,chan,stokes,i,j + 2)) && (rm(time,chan,stokes,i,j + 2) < rm(time,chan,stokes,i,j))) {
    			for (auto k = 0; k < %i; k++) if (rm(time,chan,stokes,i,k) == rm(time,chan,stokes,i,j)) rm(time,chan,stokes,i,k) = rm(time,chan,stokes,i,j + 2);
    		}
    		if ((0 != rm(time,chan,stokes,i,j)) && (0 != rm(time,chan,stokes,i,j - 2)) && (rm(time,chan,stokes,i,j - 2) < rm(time,chan,stokes,i,j))) {
    			for (auto k = 0; k < %i; k++) if (rm(time,chan,stokes,i,k) == rm(time,chan,stokes,i,j)) rm(time,chan,stokes,i,k) = rm(time,chan,stokes,i,j - 2);
    		}
    	}
    }
        """ % (grid,grid,grid,grid,), 
        {'rm':data}, 
        axis_names=('time','chan','stokes','i'), shape=(data.shape[0],data.shape[1],data.shape[2],data.shape[3]))

def computelength(islanddata,rotateddata,length,width,index,ints,nchan,ngrid,rotgrid,maxind):
    # COMPUTE vert length of non-rotated data
    BFMap("""
        for (auto i=0; i<%i; i++){
            if (isldata(time,chan,stokes,i,j)==index(ind)) length(time,chan,stokes,0,ind,j)+=1.0;
            }""" % (ngrid,),
            {'isldata':islanddata, 'length': length, 'index':index},
            axis_names = ('time','chan','stokes','j','ind'), shape=(ints,nchan,4,ngrid,maxind))
    # COMPUTE horiz length of non-rotated data
    BFMap("""
        for (auto j=0; j<%i; j++){
            if (isldata(time,chan,stokes,i,j)==index(ind)) length(time,chan,stokes,1,ind,j)+=1.0;
            }""" % (ngrid,),
            {'isldata':islanddata, 'length': length, 'index':index},
            axis_names = ('time','chan','stokes','i','ind'), shape=(ints,nchan,4,ngrid,maxind))
    # COMPUTE vert length of rotated data
    BFMap("""
        for (auto i=0; i<%i; i++){
            if (rm(time,chan,stokes,i,j)==index(ind)) length(time,chan,stokes,2,ind,j)+=1.0;
            }
        for (auto i=0; i<%i; i++){
            length(time,chan,stokes,2,ind,i) = (length(time,chan,stokes,2,ind,j)/2.0)*sqrt(2.0) ;
            }""" % (rotgrid,rotgrid,),
            {'rm':rotateddata, 'length': length,'index':index},
            axis_names = ('time','chan','stokes','j','ind'), shape=(ints,nchan,4,rotgrid,maxind))
    # COMPUTE horiz length of rotated data
    BFMap("""
        for (auto j=0; j<%i; j++){
            if (rm(time,chan,stokes,i,j)==index(ind)) length(time,chan,stokes,3,ind,j)+=1.0;
            }
        for (auto j=0; j<%i; j++){
            length(time,chan,stokes,3,ind,j) = (length(time,chan,stokes,3,ind,j)/2.0)*sqrt(2.0);
            }""" % (rotgrid,rotgrid,),
            {'rm':rotateddata, 'length': length, 'index':index},
            axis_names = ('time','chan','stokes','i','ind'), shape=(ints,nchan,4,rotgrid,maxind))
    # Take max length as length and the orthogonal length as width
    BFMap("""
         auto maxlen = 0.0;
         auto maxlenk = 0;
         auto widthplane = 0;
         auto maxwidth =0.0 ; 
         for (auto k=0; k<4; k++){
            for (auto i=0; i<%i; i++){
                if (length(time,chan,stokes,k,ind,i) > maxlen) {
                    maxlen = length(time,chan,stokes,k,ind,i);
                    maxlenk = k;
                    }
                }
            }
         length(time,chan,stokes,0,ind,0) = maxlen;
         if (maxlenk==0) widthplane = 1;
         if (maxlenk==1) widthplane = 0;
         if (maxlenk==2) widthplane = 3;
         if (maxlenk==3) widthplane = 2;
         for (auto i=0; i<%i; i++){
             if (length(time,chan,stokes,widthplane,ind,i) > maxwidth){
                 maxwidth = length(time,chan,stokes,widthplane,ind,i);
             }
             }
         width(time,chan,stokes,ind) = maxwidth;""" % (rotgrid,rotgrid,),
         {'length':length, 'width':width},
         axis_names = ('time','chan','stokes','ind'), shape=(ints,nchan,4,maxind))
