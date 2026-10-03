import { useSuspenseQuery } from "@tanstack/react-query"
import { createFileRoute } from "@tanstack/react-router"
import { CloudDownload, Plus, Search } from "lucide-react"
import { Suspense, useState } from "react"
import { ModelsService } from "@/client"
import { DataTable } from "@/components/Common/DataTable"
import { AddModel } from "@/components/Models/AddModel"
import { columns } from "@/components/Models/columns"
import { SearchHuggingFace } from "@/components/Models/SearchHuggingFace"
import PendingModels from "@/components/Pending/PendingModels"
import { Button } from "@/components/ui/button"

function getModelsQueryOptions() {
  return {
    queryFn: async () =>
      (await ModelsService.readModels({ query: { skip: 0, limit: 100 } })).data,
    queryKey: ["models"],
  }
}

export const Route = createFileRoute("/_layout/models")({
  component: Models,
  head: () => ({
    meta: [
      {
        title: "Models - Inference Matrix",
      },
    ],
  }),
})

function ModelsTableContent() {
  const { data: models } = useSuspenseQuery(getModelsQueryOptions())

  if (!models || models.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center text-center py-12">
        <div className="rounded-full bg-muted p-4 mb-4">
          <Search className="h-8 w-8 text-muted-foreground" />
        </div>
        <h3 className="text-lg font-semibold">You don't have any models yet</h3>
        <p className="text-muted-foreground">Add a new model to get started</p>
      </div>
    )
  }

  return <DataTable columns={columns} data={models} />
}

function ModelsTable() {
  return (
    <Suspense fallback={<PendingModels />}>
      <ModelsTableContent />
    </Suspense>
  )
}

function Models() {
  const [isAddModelOpen, setIsAddModelOpen] = useState(false)
  const [isHuggingFaceSearchOpen, setIsHuggingFaceSearchOpen] = useState(false)

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Models</h1>
          <p className="text-muted-foreground">
            Manage your GGUF models for inference
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={() => setIsHuggingFaceSearchOpen(true)}
          >
            <CloudDownload className="mr-2 h-4 w-4" />
            Search HuggingFace
          </Button>
          <Button onClick={() => setIsAddModelOpen(true)}>
            <Plus className="mr-2 h-4 w-4" />
            Add Model
          </Button>
        </div>
      </div>
      <ModelsTable />
      <AddModel
        isOpen={isAddModelOpen}
        onClose={() => setIsAddModelOpen(false)}
      />
      <SearchHuggingFace
        isOpen={isHuggingFaceSearchOpen}
        onClose={() => setIsHuggingFaceSearchOpen(false)}
      />
    </div>
  )
}
