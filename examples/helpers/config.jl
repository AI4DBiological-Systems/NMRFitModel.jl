@with_kw struct ResonancePlotConfigType

    save_folder_path = "./plots/resonance_groups"
    postfunc::Function = real
    size_inches = (8/2.54, 4/2.54)

    title_font_size = 30
    width_padding_proportion = 0.0

    reverse_x_axis = false
    grid_visible = true
    background_color = :white

    line_width = 3.0

    x_label = "ppm"
    x_tick_decimal_places::Int = 2
    x_tick_label_size = 15
    x_label_font_size = 20

    
    y_label = "Real part of spectrum"
    y_tick_decimal_places::Int = 2
    y_tick_label_size = 15
    y_label_font_size = 20

    use_Wilkinson_for_ticks = false # default is LinearTicks from Makie.

    legend_font_size = 15
    legend_orientation = :horizontal # :vertical or :horizontal
    legend_frame_visible = false
    legend_line_width = 2.0
    
    pt_per_inch::Int = 72
    misc_font_size = 12
    resize_before_saving::Bool = true
end
