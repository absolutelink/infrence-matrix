import { zodResolver } from "@hookform/resolvers/zod"
import { useForm } from "react-hook-form"
import { z } from "zod"

import {
  type Model,
  ModelsService,
} from "@/client"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form"
import { Input } from "@/components/ui/input"
import { LoadingButton } from "@/components/ui/loading-button"
import useCustomToast from "@/hooks/useCustomToast"
import { useQueryClient } from "@tanstack/react-query"
import { useEffect } from "react"

const formSchema = z.object({
  name: z.string().min(1, { message: "Name is required" }),
  path: z.string().min(1, { message: "Path is required" }),
  size_bytes: z.coerce.number().min(0, { message: "Size must be positive" }),
  architecture: z.string().min(1, { message: "Architecture is required" }),
  parameter_count: z.coerce.number().optional(),
  quantization: z.string().min(1, { message: "Quantization is required" }),
  supports_embeddings: z.boolean().default(false),
  supports_vision: z.boolean().default(false),
  context_length: z.coerce.number().min(1, { message: "Context length is required" }),
  license: z.string().optional(),
  tags: z.array(z.string()).default([]),
  description: z.string().optional(),
  source: z.string().min(1, { message: "Source is required" }),
  source_repo_id: z.string().optional(),
  source_url: z.string().url().optional().or(z.literal("")),
  source_file: z.string().optional(),
})

type FormData = z.infer<typeof formSchema>

interface EditModelProps {
  isOpen: boolean
  onClose: () => void
  model: Model
}

export default function EditModel({ isOpen, onClose, model }: EditModelProps) {
  const queryClient = useQueryClient()
  const { showSuccessToast } = useCustomToast()

  const form = useForm({
    resolver: zodResolver(formSchema),
    mode: "onBlur",
    defaultValues: {
      name: "",
      path: "",
      size_bytes: 0,
      architecture: "",
      parameter_count: undefined,
      quantization: "",
      supports_embeddings: false,
      supports_vision: false,
      context_length: 4096,
      license: "",
      tags: [],
      description: "",
      source: "",
      source_repo_id: "",
      source_url: "",
      source_file: "",
    },
  })

  useEffect(() => {
    if (model) {
      form.reset({
        name: model.name,
        path: model.path,
        size_bytes: model.size_bytes,
        architecture: model.architecture,
        parameter_count: model.parameter_count ?? undefined,
        quantization: model.quantization,
        supports_embeddings: model.supports_embeddings,
        supports_vision: model.supports_vision,
        context_length: model.context_length,
        license: model.license ?? "",
        tags: model.tags ?? [],
        description: model.description ?? "",
        source: model.source,
        source_repo_id: model.source_repo_id ?? "",
        source_url: model.source_url ?? "",
        source_file: model.source_file ?? "",
      })
    }
  }, [model, form])

  const onSubmit = async (data: FormData) => {
    try {
      await ModelsService.updateModel({
        path: { id: model.id! },
        body: data as any,
      })
      showSuccessToast("Model updated successfully")
      onClose()
      queryClient.invalidateQueries({ queryKey: ["models"] })
    } catch (error: any) {
      console.error(error)
    }
  }

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Edit Model</DialogTitle>
          <DialogDescription>
            Update model information
          </DialogDescription>
        </DialogHeader>

        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <FormField
                control={form.control}
                name="name"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Model Name</FormLabel>
                    <FormControl>
                      <Input {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="architecture"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Architecture</FormLabel>
                    <FormControl>
                      <Input {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>

            <FormField
              control={form.control}
              name="path"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>File Path</FormLabel>
                  <FormControl>
                    <Input {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <div className="grid grid-cols-3 gap-4">
              <FormField
                control={form.control}
                name="size_bytes"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Size (bytes)</FormLabel>
                    <FormControl>
                      <Input type="number" {...field} value={typeof field.value === 'number' ? field.value : ''} onChange={(e) => field.onChange(e.target.valueAsNumber)} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="parameter_count"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Parameters</FormLabel>
                    <FormControl>
                      <Input type="number" {...field} value={typeof field.value === 'number' ? field.value : ''} onChange={(e) => field.onChange(e.target.valueAsNumber)} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="context_length"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Context Length</FormLabel>
                    <FormControl>
                      <Input type="number" {...field} value={typeof field.value === 'number' ? field.value : ''} onChange={(e) => field.onChange(e.target.valueAsNumber)} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>

            <FormField
              control={form.control}
              name="quantization"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Quantization</FormLabel>
                  <FormControl>
                    <Input {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <div className="flex justify-end gap-2 pt-4">
              <LoadingButton type="button" variant="outline" onClick={onClose}>
                Cancel
              </LoadingButton>
              <LoadingButton type="submit">
                Update Model
              </LoadingButton>
            </div>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  )
}
