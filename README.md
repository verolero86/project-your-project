# project-your-project

`project-your-project` is a simple tool to visualize your project's progress.
`project-your-project` generates a Gannt chart that displays both the baseline
and current schedule to easily visualize if your project is projected to be on
track or not.

This project uses the excellent tutorial as the base and adds customizations:

https://towardsdatascience.com/gantt-charts-with-pythons-matplotlib-395b7af72d72

## Usage

The code currently requires the following arguments:

```
    -i INPUTFILE 
    -t TITLE 
    -o OUTFILEPRE 
    -c CATEGORIES [CATEGORIES ...] 
    -x HEXCOLORS [HEXCOLORS ...] 
    -b BARHEIGHTS [BARHEIGHTS ...]
```

### Example

An example using a timeline with four categories:

`./project-your-project.py -i my_project_schedule.xlsx -t "Timeline Title" -o ./timeline_ -c "One Category" "Another Category" "Category Z" "Baseline" -x "#000000" "#ff0000" "#00ff00" "#0000ff" -b 0.8 0.8 0.8 0.3`

## Author 

[@verolero86](https://github.com/verolero86)


