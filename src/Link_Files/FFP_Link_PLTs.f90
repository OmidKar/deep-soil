!  FFP_Link_PLTs.f90 
!
!
!****************************************************************************
!
!  PROGRAM: FFP_Link_PLTs
!
!  PURPOSE:  Links all 223 snac.plt files into a .csv file
!  Majid Nazem July 2019
!
!****************************************************************************

program FFP_Link_PLTs
    implicit none

    ! Variables and constants
    Integer, Parameter :: N_Samples = 223
    Integer, Parameter :: Input_unit = 12
    Integer, Parameter :: Output_unit = 13

    Character (20) :: Filename
    Character (50) :: Dirname
    Integer :: i, j
    Integer :: Archive (n_Samples)
    
    Data (Archive(i), I=1,N_Samples)/1,3,13,16,19,20,22,24,26,27,29,30,32,33,36,38,40,50,53,56,57,61,62,63,64,65,66,67,70,71,72,73,74,75,77,79,88,90,93,95,101,103,104,106,107,108,109,110,112,113,114,115,116,119,122,125,126,127,128,129,130,131,132,133,134,136,137,138,139,140,141,142,143,144,146,147,148,149,150,151,152,153,154,156,159,161,162,163,164,165,166,167,168,169,170,171,175,176,177,178,179,180,181,183,184,186,187,188,189,190,191,193,196,198,199,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,226,230,233,235,237,238,239,240,241,242,243,245,246,248,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,267,268,270,271,272,273,274,275,276,277,278,279,280,281,282,283,285,287,288,289,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,307,308,310,313,314,315,317,318,319,320,322,325,326,327,328,329,330,331,332,333/
    Real(8) :: Disp, Force, Velocity
    
    ! Open Output file
    Open (Output_unit, File = "Disp_Force_Vel.csv", Status = "Unknown")
    Write (Output_unit, *) "Archive No, Penetration/D, Force, Velocity"
    
    ! Main loop over all samples
    Filename = "SNAC.PLT"
    dirname = ""
    Do i=1, N_Samples
        ! Get the dir name
        j = Archive(i)
        If (j <= 9)Then
           Write (dirname,'("snac",i1)') j
        Else If (j <= 99)Then
           Write (dirname,'("snac",i2)') j
        Else If (j <= 999)Then
           Write (dirname,'("snac",i3)') j
        Else If (j <= 9999)Then
           Write (dirname,'("snac",i4)') j
        Endif
        dirname = "copy /y " // Trim(dirname) // "\snac.plt snac.plt"
        Call System (dirname)
        Open (Input_unit,FILE=Filename,STATUS='OLD')
        
        ! Read the file and add all records to output file
        Do
            If (Eof(Input_unit)) Then
                Exit 
            End If
            Read (Input_unit,*) Disp, Force, Velocity
            Write(Output_unit,'(I4,3(1A,F20.12))') j, ",", Disp, ",", Force, ",", Velocity
        End Do
        
        Close (Input_unit)
    End Do
    
    Write (*,*) "Job completed! Press any key to continue..."
    Pause
    Close(Output_unit)

end program FFP_Link_PLTs

