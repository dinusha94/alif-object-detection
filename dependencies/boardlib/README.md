# BoardLib

BOARD support library for Alif Semiconductor Development and Application Kits

## Development Boards Supported

This board library has been tested on the following evaluation platforms.
- Generation 1 Development Kit
- Generation 1 AI/ML Application Kit (Alpha1 and Alpha2 revisions)
- Generation 2 Development Kit
- Generation 2 AI/ML Application Kit

For most of the application projects, the board library is a submodule and will be part of the project once *submodule init* and *submodule update* is done.

To manually add the board library to a project please follow the instructions below.

## Manually Adding boardlib to a project in VSCode
Most of the application examples provided by Alif Semiconductor already have support for boardlib integrated as a Git submodule. Depending on the board the 
project is built for, the appropriate board will have to be selected in the board.h file in either the common or app folder.
For adding boardlib support to your project as a submodule
1. Run *git submodule add https://github.com/alifsemi/alif_boardlib.git "**Optional** path with name of the folder you want to add submodule to"*
2. Copy board_template.h from the newly created alif_boardlib folder to your application folder as board.h
3. Edit the board.h to enable any of the Alif Semiconductor board you would like to use.
4. If creating a version of your board, make a copy of any of the existing Alif Semiconductor board folder closest to your design and make appropriate changes in the boards_def.h and board_init.c files and add your board as an option to the board.h file.


## Manually Adding boardlib to a project in Arm-DS

1. Select the project in the left-hand project explorer, and right-click.
2. Select New -> Folder
3. Click Advanced >> and then select the 3rd option: Link to alternate location
4. Click Browse... and then select the 'boardlib_DEV' folder that you cloned
5. Rename 'Folder name:' to simply 'board' and click Finish
6. Copy board_template.h from 'board' folder to the root of the project
7. Rename the board_template.h to board.h
8. Open and change #if 0 to 1 and change BOARD_ALIF_DEVKIT_VARIANT to match your board
9. Add the below include folder paths in Project properties
    1. /${ProjName}/
    2. /${ProjName}/board
