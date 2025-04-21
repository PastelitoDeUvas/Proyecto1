library(shiny)
library(readxl)
library(knitr)
library(kableExtra)
library(ggplot2)
library(caret)
library(shinyWidgets)

# Cargar datos
data <- read_xlsx("C:\\Users\\JUAN\\Desktop\\R\\Proyecto 2\\Datos_Rotación.xlsx")

# Crear un data frame con las variables, categorías e hipótesis
variables <- data.frame(
  Variable = c("Distancia_Casa", "Años_Última_Promoción", "Equilibrio_Trabajo_Vida", 
               "Género", "Estado_Civil", "Horas_Extra"),
  Categoria = c("Cuantitativa", "Cuantitativa", "Cuantitativa", 
                "Categórica", "Categórica", "Categórica"),
  Hipotesis = c(
    "Se espera que empleados que viven más lejos tengan mayor rotación debido al tiempo y costos de desplazamiento.",
    "Se espera que empleados que llevan muchos años sin promoción busquen nuevas oportunidades fuera de la empresa.",
    "Un mal equilibrio entre trabajo y vida personal puede aumentar la probabilidad de rotación.",
    "El género podría influir en la rotación debido a diferencias en oportunidades o cargas laborales percibidas.",
    "El estado civil puede afectar la rotación, por ejemplo, empleados solteros pueden cambiar más de trabajo que los casados.",
    "Se espera que empleados que hacen horas extra frecuentes tengan mayor rotación por estrés y agotamiento."
  )
)

# UI
ui <- fluidPage(
  setBackgroundColor("#FFEBF6"), # Fondo pastel rosado
  titlePanel("✨ Predicción de Rotación de Empleados ✨"),
  sidebarLayout(
    sidebarPanel(
      fileInput("file", "📂 Cargar datos de empleados", accept = ".xlsx"),
      actionButton("predict", "🔮 Predecir Rotación", class = "btn-primary")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("📖 Informe", 
                 h3("Introducción"),
                 p("La rotación de empleados es un factor clave en la gestión del talento dentro de una empresa, ya que impacta directamente en la productividad, los costos operativos y el clima organizacional. Comprender las variables que influyen en la decisión de un empleado de abandonar la empresa permite desarrollar estrategias para reducir la rotación y mejorar la retención del talento."),
                 p("En este informe, se presenta un modelo para medir la rotación de empleados a partir de seis variables explicativas: tres cuantitativas y tres categóricas. El objetivo es identificar los factores más relevantes que influyen en la permanencia o salida de los trabajadores y evaluar su impacto a través de un modelo estadístico."),
                 p("Para ello, se emplea un enfoque basado en análisis de datos y técnicas de modelado, con el fin de proporcionar una herramienta que ayude a la empresa a tomar decisiones informadas en la gestión de su capital humano."),
                 h3("📊 Variables del Modelo"),
                 tableOutput("variablesTable")
        ),
        tabPanel("📊 Gráficos", plotOutput("plot")),
        tabPanel("📜 Resultados", tableOutput("results"))
      )
    )
  )
)

# Server
server <- function(input, output) {
  data <- reactive({
    req(input$file)
    read_excel(input$file$datapath)
  })
  
  output$plot <- renderPlot({
    req(data())
    ggplot(data(), aes(x = Edad, y = Salario, color = factor(Rotacion))) +
      geom_point(size = 3) +
      theme_minimal() +
      scale_color_manual(values = c("#FF69B4", "#87CEFA")) +
      labs(title = "📊 Empleados y Rotación", x = "Edad", y = "Salario", color = "Rotación")
  })
  
  output$results <- renderTable({
    req(data())
    data()
  })
  
  output$variablesTable <- renderUI({
    variables %>%
      kable("html", col.names = c("Variable", "Categoría", "Hipótesis"), align = "l") %>%
      kable_styling(full_width = FALSE, bootstrap_options = c("striped", "hover", "condensed"), 
                    font_size = 16, position = "center", 
                    row_label_position = "c") %>%
      column_spec(1, bold = TRUE, color = "#D63384") %>%
      column_spec(2, color = "#6A0572") %>%
      column_spec(3, width = "50em")
  })
}

# Ejecutar App
shinyApp(ui, server)