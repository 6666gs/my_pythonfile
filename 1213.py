import gdsfactory as gf

@gf.cell
def gc(period: float = 1,
       n: int = 20,
       duty_ratio: float = 0.5,
       angle: float = 10,
       length_taper: float = 40,
       length_end: float = 50,
       width_end: float = 0.9,
       ) -> gf.Component:
    c = gf.Component()
    secs_teeth = []
    for i in range(n):
        secs_teeth.append(gf.Section(width=period * duty_ratio, offset=(i - duty_ratio / 2 + 1) * period, layer="WG"))
    secs_teeth = tuple(secs_teeth)
    cs_teeth = gf.cross_section.cross_section(width=length_taper, offset=-length_taper / 2, layer="WG",
                                              sections=secs_teeth)
    p_teeth = gf.path.arc(radius=length_taper, angle=angle, start_angle=180-angle / 2)
    ref_teeth = c << p_teeth.extrude(cross_section=cs_teeth)
    end = gf.components.straight(length=length_end + length_taper / 2, layer="WG", width=width_end)
    ref_end = c << end
    ref_teeth.xmax = ref_end.xmin + length_taper / 2
    ref_teeth.y = ref_end.y
    port = ref_end['o2']
    port.name = 'o1'
    c.add_port(port=port)

    return c

d = gc()



d.show(gdspath="grating_coupling3.gds")