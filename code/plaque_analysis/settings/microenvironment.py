"""Gene panels, fixed class identities and display conventions."""

CLASS_ORDER = ("S2", "S1", "S3")
GLIAL_ORDER = CLASS_ORDER
COLORS = {"S2": "#387EA6", "S1": "#C89530", "S3": "#9A557C"}
PRIMARY = ("REACTOME_NEURONAL_SYSTEM_UCell", "HALLMARK_UNFOLDED_PROTEIN_RESPONSE_UCell")
PROGRAM_LABELS = dict(zip(PRIMARY, ("Neuronal system", "Unfolded protein response")))
GENES = (("Hspa5", "neurons"), ("Gfap", "astrocytes"), ("P2ry12", "microglia"), ("Trem2", "microglia"))
GLIAL_PROGRAMS = (("DAM_KEREN_SHAUL_2017_UCell", "Microglia", "DAM gene set", 34),
                  ("DAA_HABIB_2020_UCell", "Astrocyte", "DAA gene set", 35))
NAMED_TYPES = ("Excitatory neuron", "Inhibitory neuron", "Astrocyte", "Microglia", "Oligodendrocyte")
DISPLAY_TYPES = (*NAMED_TYPES, "Other / unresolved")
TYPE_LABELS = ("Excitatory", "Inhibitory", "Astrocytes", "Microglia", "Oligodendro.", "Other")
INK = "#253039"
GRID = "#E3E7EA"
