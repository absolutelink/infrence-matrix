import { useSuspenseQuery } from "@tanstack/react-query"
import { createFileRoute } from "@tanstack/react-router"
import { FileText, Upload } from "lucide-react"
import { Suspense, useState } from "react"
import { V1FilesService } from "@/client"
import { DataTable } from "@/components/Common/DataTable"
import { columns } from "@/components/Files/columns"
import UploadFilesDialog from "@/components/Files/UploadFilesDialog"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"

function getFilesQueryOptions() {
  return {
    queryFn: async () => (await V1FilesService.v1.listFiles()).data,
    queryKey: ["files"],
  }
}

export const Route = createFileRoute("/_layout/files")({
  component: Files,
  head: () => ({
    meta: [{ title: "Files - Inference Matrix" }],
  }),
})

function FilesTableContent() {
  const { data } = useSuspenseQuery(getFilesQueryOptions())
  const files = data?.data ?? []

  if (files.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center text-center py-12">
        <div className="rounded-full bg-muted p-4 mb-4">
          <FileText className="h-8 w-8 text-muted-foreground" />
        </div>
        <h3 className="text-lg font-semibold">No files uploaded yet</h3>
        <p className="text-muted-foreground">Upload a file to get started</p>
      </div>
    )
  }

  return <DataTable columns={columns} data={files} />
}

function FilesTable() {
  return (
    <Suspense
      fallback={
        <div className="space-y-2">
          <Skeleton className="h-12 w-full" />
          <Skeleton className="h-12 w-full" />
          <Skeleton className="h-12 w-full" />
        </div>
      }
    >
      <FilesTableContent />
    </Suspense>
  )
}

function Files() {
  const [isUploadOpen, setIsUploadOpen] = useState(false)

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Files</h1>
          <p className="text-muted-foreground">
            Manage uploaded files for batch, vision, and other purposes
          </p>
        </div>
        <Button onClick={() => setIsUploadOpen(true)}>
          <Upload className="mr-2 h-4 w-4" />
          Upload Files
        </Button>
      </div>
      <FilesTable />
      <UploadFilesDialog
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
      />
    </div>
  )
}
