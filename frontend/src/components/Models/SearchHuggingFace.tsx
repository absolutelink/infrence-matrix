import { useState, useEffect } from "react"
import { Search, Download } from "lucide-react"
import { useQuery, useQueryClient } from "@tanstack/react-query"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import useCustomToast from "@/hooks/useCustomToast"
import { AddModelFromHF } from "./AddModelFromHF"

interface HuggingFaceModel {
  id: string
  modelId: string
  author: string
  downloads: number
  likes: number
  tags: string[]
  createdAt?: string
}

interface SearchHuggingFaceProps {
  isOpen: boolean
  onClose: () => void
}

const searchModels = async (query: string): Promise<HuggingFaceModel[]> => {
  const token = localStorage.getItem("access_token")
  const baseUrl = import.meta.env.VITE_API_URL || "http://localhost:8000"
  const response = await fetch(
    `${baseUrl}/api/v1/huggingface/search?search=${encodeURIComponent(query)}&limit=20`,
    {
      headers: {
        "Authorization": `Bearer ${token}`,
      },
    }
  )
  if (!response.ok) {
    throw new Error("Failed to search HuggingFace")
  }
  return response.json()
}

export function SearchHuggingFace({ isOpen, onClose }: SearchHuggingFaceProps) {
  const [searchQuery, setSearchQuery] = useState("")
  const [selectedModel, setSelectedModel] = useState<HuggingFaceModel | null>(null)
  const { showErrorToast } = useCustomToast()

  const { data: searchResults, isLoading } = useQuery({
    queryKey: ["hf-search", searchQuery],
    queryFn: () => searchModels(searchQuery),
    enabled: searchQuery.length >= 2,
    staleTime: 5 * 60 * 1000,
  })

  const handleSelectModel = (model: HuggingFaceModel) => {
    setSelectedModel(model)
  }

  return (
    <>
      <Dialog open={isOpen} onOpenChange={(open) => {
        if (!open) {
          onClose()
          setSearchQuery("")
        }
      }}>
        <DialogContent className="max-w-4xl max-h-[80vh] flex flex-col">
          <DialogHeader>
            <DialogTitle>Search HuggingFace Models</DialogTitle>
            <DialogDescription>
              Search for GGUF models on HuggingFace and add them to your inference matrix
            </DialogDescription>
          </DialogHeader>

          <div className="flex items-center gap-2 mb-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search models (e.g., llama-2, mistral, phi-3)..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>
          </div>

          <div className="flex-1 overflow-auto">
            {isLoading ? (
              <div className="space-y-2">
                {Array.from({ length: 5 }).map((_, i) => (
                  <Skeleton key={i} className="h-20 w-full" />
                ))}
              </div>
            ) : searchResults && searchResults.length > 0 ? (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Model</TableHead>
                    <TableHead>Downloads</TableHead>
                    <TableHead>Likes</TableHead>
                    <TableHead>Tags</TableHead>
                    <TableHead>
                      <span className="sr-only">Actions</span>
                    </TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {searchResults.map((model) => (
                    <TableRow 
                      key={model.id}
                      className="cursor-pointer hover:bg-muted/50"
                      onClick={() => handleSelectModel(model)}
                    >
                      <TableCell>
                        <div className="font-medium">{model.modelId || model.id}</div>
                        <div className="text-sm text-muted-foreground">
                          by {model.author}
                        </div>
                      </TableCell>
                      <TableCell>
                        {model.downloads?.toLocaleString() || "-"}
                      </TableCell>
                      <TableCell>
                        {model.likes?.toLocaleString() || "-"}
                      </TableCell>
                      <TableCell>
                        <div className="flex flex-wrap gap-1">
                          {model.tags?.slice(0, 3).map((tag: string) => (
                            <Badge key={tag} variant="secondary" className="text-xs">
                              {tag}
                            </Badge>
                          ))}
                          {model.tags?.length > 3 && (
                            <Badge variant="outline" className="text-xs">
                              +{model.tags.length - 3}
                            </Badge>
                          )}
                        </div>
                      </TableCell>
                      <TableCell onClick={(e) => e.stopPropagation()}>
                        <Button
                          size="sm"
                          onClick={() => handleSelectModel(model)}
                        >
                          <Download className="h-4 w-4 mr-1" />
                          Select
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            ) : searchQuery.length >= 2 ? (
              <div className="text-center py-8 text-muted-foreground">
                No models found. Try a different search term.
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                Enter at least 2 characters to search
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>

      {selectedModel && (
        <AddModelFromHF
          model={selectedModel}
          onClose={() => setSelectedModel(null)}
        />
      )}
    </>
  )
}
